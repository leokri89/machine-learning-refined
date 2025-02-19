print('Loading function sofia ai assistant - data ingestion')

## loading libraries
import os
import sys

if "LAMBDA_TASK_ROOT" in os.environ:
    envLambdaTaskRoot = os.environ["LAMBDA_TASK_ROOT"]
    sys.path.insert(0, "/var/lang/lib/python3.8/site-packages")


import logging  
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# langchain
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.text_splitter import SpacyTextSplitter
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy

# other libraries
import csv
import json
import numpy as np
from botocore.exceptions import ClientError

# para bm25
from rank_bm25 import BM25Okapi
import nltk
import unicodedata
import re
import pandas as pd

# Amazon
import boto3

#Sql engine
from sqlalchemy import create_engine

# NLP
import spacy
import dill as pickle
from io import BytesIO
from io import StringIO
from botocore.client import Config
from datetime import datetime, timezone, timedelta

nltk.data.path.append('/usr/share/nltk_data')


def get_secret(secret_name,region_name):
    # Create a Secrets Manager client with custom endpoint
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        raise e

    return get_secret_value_response['SecretString']


def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    
    return lemmatized_sentence


def read_csv_file_spacy_chunk_s3(bucket_name, filename, text_splitter, delim='|', chunk_size=0):
    print('Starting read_csv_file_spacy_chunk')
    
    # Inicializa o cliente S3
    s3 = boto3.client('s3')
    
    # Baixa o arquivo do S3
    s3_object = s3.get_object(Bucket=bucket_name, Key=filename)
    file_content = s3_object['Body'].read().decode('utf-8')
    
    # Usa StringIO para simular um arquivo em memória
    file = StringIO(file_content)
    
    # Lê o arquivo CSV
    csv_reader = csv.DictReader(file, delimiter=delim)

    descricao_list = []

    count = 0
    for row in csv_reader:
        count += 1

        docs = text_splitter.split_text(row['resposta'])
        
        for d_ in docs:
            doc = Document(page_content=row['pergunta'] + ' ' + d_, metadata={'source': row['link'], 'title': row['pergunta'], 'id': row['qid']})
            descricao_list.append(doc)

    print(f'Arquivo {filename} carregado com sucesso do bucket {bucket_name}!')
    return descricao_list


def strip_accents(text):
    """Strip accents and punctuation from text. 
    For instance: strip_accents("João e Maria, não entrem!") 
    will return "Joao e Maria  nao entrem "

    Parameters:
    text (str): Input text

    Returns:
    str: text without accents and punctuation

    """    
    nfkd = unicodedata.normalize('NFKD', text)
    newText = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    return re.sub('[^a-zA-Z0-9 \\\']', ' ', newText)


def preprocess_string(txt, remove_stop=True, do_stem=True, to_lower=True, do_lemma = False):
    """
    Return a preprocessed tokenized text.
    
    Args:
        txt (str): original text to process
        remove_stop (boolean): to remove or not stop words (common words)
        do_stem (boolean): to do or not stemming (suffixes and prefixes removal)
        to_lower (boolean): remove or not capital letters.
        
    Returns:
        Return a preprocessed tokenized text.
    """      

    if to_lower:
        txt = txt.lower()

    # antes de tirar ascento
    if do_lemma:
        txt = lemmatize_sentence(txt)

    txt = strip_accents(txt)
            
    tokens = nltk.tokenize.word_tokenize(txt, language="portuguese")
    
    if remove_stop:
        tokens = [tk for tk in tokens if tk not in stop_words]

    if do_stem:
        tokens = [stemmer.stem(tk) for tk in tokens]

    return tokens


def create_langchain_vector_embedding_using_bedrock(bedrock_client):
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=MODEL_ID_EMB)
    return bedrock_embeddings_client


def insert_record(table_name, chat_id, tenant, status, status_message):
    # Inicializar o cliente DynamoDB
    dynamodb = boto3.client('dynamodb')

        # Obter a data e hora atuais no fuso horário UTC-3
    utc_minus_3 = timezone(timedelta(hours=-3))
    current_time = datetime.now(utc_minus_3).strftime('%Y-%m-%d %H:%M:%S')
    
    # Estruturar o item a ser inserido
    item = {
        'chat_id': {'S': chat_id},
        'tenant': {'S': tenant},
        'status': {'S': status},
        'statusTimestamp': {'S': current_time},
        'status_message':{'S':status_message}
    }

    try:
        # Inserir o item na tabela
        response = dynamodb.put_item(
            TableName=table_name,
            Item=item
        )
        return response
    except ClientError as e:
        # Tratar exceções
        print(f"Erro ao inserir item: {e.response['Error']['Message']}")
        return None
        

def handler(event, context):
    print("event", event)
    print("event_type", type(event))
    print("event_keys", event.keys())

    secret_name = os.environ.get('SECRET_NAME')
    region_name = os.environ.get('REGION_NAME')
    pgvector_host = os.environ.get('PGVECTOR_HOST')
    s3_bucket_name_dados = os.environ.get('S3_BUCKET_NAME_DADOS')
    s3_bucket_name_artefatos = os.environ.get('S3_BUCKET_NAME_ARTEFATOS')
    dynamo_table_name =os.environ.get('DYNAMO_TABLE_NAME')

    # Extrair dados do payload
    tenant_id = event.get('tenant')
    id_chat = event.get('chat_id')
    
    # Verificação básica dos dados recebidos
    if not tenant_id or not id_chat:
        return {
            'statusCode': 400,
            'body': json.dumps('Invalid input: tenant and chat_id are required.')
        }

    # setup parameters
    MODEL_LIST = {"anthropic":"anthropic.claude-3-haiku-20240307-v1:0",
                  "emb-titan-v1": "amazon.titan-embed-g1-text-02"
                 }

    MODEL_ID_EMB = MODEL_LIST['emb-titan-v1']
    # MODEL_ID_LLM = MODEL_LIST['anthropic']
    
    secrets_db=json.loads(get_secret(secret_name,region_name))
    print("get data secrets")

    user_db_=secrets_db["username"]
    print(user_db_)
    password_db_=secrets_db["password"]

    PGVECTOR_DRIVER="psycopg2"
    PGVECTOR_USER=user_db_
    PGVECTOR_PASSWORD=password_db_
    PGVECTOR_HOST=pgvector_host
    PGVECTOR_PORT=5432
    PGVECTOR_DATABASE="postgres"
    COLLECTION_NAME=id_chat
    DISTANCE_STRATEGY="DistanceStrategy.EUCLIDEAN"

    # setup bedrock
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    boto3_bedrock = boto3.client('bedrock-runtime')

    #setup nltk
    stop_words = ['a', 'à', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'às', 'até',
    'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois', 'do', 'dos', 'e', 'é', 'ela',
    'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram', 'éramos', 'essa', 'essas', 'esse', 'esses', 'esta',
    'está', 'estamos', 'estão', 'estar', 'estas', 'estava', 'estavam', 'estávamos', 'este', 'esteja', 'estejam',
    'estejamos', 'estes', 'esteve', 'estive', 'estivemos', 'estiver', 'estivera', 'estiveram', 'estivéramos', 'estiverem',
    'estivermos', 'estivesse', 'estivessem', 'estivéssemos', 'estou', 'eu', 'foi', 'fomos', 'for', 'fora', 'foram',
    'fôramos', 'forem', 'formos', 'fosse', 'fossem', 'fôssemos', 'fui', 'há', 'haja', 'hajam', 'hajamos', 'hão',
    'havemos', 'haver', 'hei', 'houve', 'houvemos', 'houver', 'houvera', 'houverá', 'houveram', 'houvéramos',
    'houverão', 'houverei', 'houverem', 'houveremos', 'houveria', 'houveriam', 'houveríamos', 'houvermos',
    'houvesse', 'houvessem', 'houvéssemos', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mas', 'me', 'mesmo', 'meu', 'meus',
    'minha', 'minhas', 'muito', 'na', 'não', 'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num',
    'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'são',
    'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'seu',
    'seus', 'só', 'somos', 'sou', 'sua', 'suas', 'também',
    'te', 'tem', 'tém', 'temos', 'tenha', 'tenham', 'tenhamos', 'tenho', 'terá', 'terão', 'terei', 'teremos', 'teria', 'teriam', 'teríamos',
    'teu', 'teus', 'teve', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem',
    'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês',
    'vos']

    print("load spacy nlp") 
    stemmer = nltk.stem.PorterStemmer()
    nlp = spacy.load('en_core_web_sm')
    print("load spacy nlp ok ") 

    # Continuação do processo
    print("bedrock initialize") 
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock(boto3_bedrock)    
    text_chunksize=2500
    text_chunkover=10t

    print("Prepare SpacyTextSplitter") 
    BASE_DADOS=f"{tenant_id}/{id_chat}/data/chat_data.csv"
    print(BASE_DADOS)
    text_splitter = SpacyTextSplitter(chunk_size=text_chunksize,chunk_overlap=(int(text_chunkover)/100)*text_chunksize)
    docs = read_csv_file_spacy_chunk_s3(s3_bucket_name_dados,BASE_DADOS, text_splitter,delim=',')


    # Construct the connection string to the PostgreSQL database
    CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
                driver=PGVECTOR_DRIVER,
                user=PGVECTOR_USER,
                password=PGVECTOR_PASSWORD,
                host=PGVECTOR_HOST,
                port=PGVECTOR_PORT,
                database=PGVECTOR_DATABASE,
            )

    store = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=bedrock_embeddings_client,
        distance_strategy=DISTANCE_STRATEGY # Douglas: DistanceStrategy.COSINE ? 
    )
    
    print("PGVector from_documents")
    db = PGVector.from_documents(
        documents=docs,
        embedding=bedrock_embeddings_client,
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,
    )

    print("read_csv corpus_")
    # Reading corpus files
    s3_client = boto3.client('s3')
    model_version = 1
    object_key_bm25 = f"{tenant_id}/{id_chat}/{model_version}/bm25.pickle"
    corpus_object_key = BASE_DADOS
    
    print("arquivo de entrada")
    print(corpus_object_key)

    corpus_obj = s3_client.get_object(Bucket=s3_bucket_name_dados, Key=corpus_object_key)
    corpus_ = pd.read_csv(corpus_obj['Body'],delimiter=',')

    print("generate bm25 pickle")
    # Reading corpus files
    corpus_list_ = list(corpus_['resposta'])
    tokenized_corpus = [preprocess_string(doc,True,True) for doc in corpus_list_]
    bm25 = BM25Okapi(tokenized_corpus, k1=0.1, b=0.8)
    corpus_.reset_index(inplace=True)
    
    print("Cria um objeto BytesIO para armazenar os dados do pickle")
    with BytesIO() as data:
        # Dump do objeto bm25 para o BytesIO
        pickle.dump(bm25, data)
        data.seek(0)  # Volta ao início do arquivo
        print("Arquivo pickle criado! Iniciando upload...")

        # Faz o upload do arquivo para o S3
        s3_client.upload_fileobj(data, s3_bucket_name_artefatos, object_key_bm25)

    print(f'Arquivo {object_key_bm25} copiado com sucesso para o bucket {s3_bucket_name_artefatos} no S3.')

    table_name = dynamo_table_name
    chat_id = id_chat
    tenant = tenant_id
    status = 'inservice'
    status_message='chat created successfully'
    
    response = insert_record(table_name, chat_id, tenant, status, status_message)
    if response:
        print("Item inserido com sucesso.")
    else:
        print("Falha ao inserir item.")

    return "ok"
