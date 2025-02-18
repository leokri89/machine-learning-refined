print('Loading function sofia ai assistant - get answer')

## loading libraries
import os
import sys


if "LAMBDA_TASK_ROOT" in os.environ:
    envLambdaTaskRoot = os.environ["LAMBDA_TASK_ROOT"]
    sys.path.insert(0, "/var/lang/lib/python3.8/site-packages")

import boto3

import logging  
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# langchain
import langchain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.docstore.document import Document
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.vectorstores.pgvector import PGVector, DistanceStrategy
from langchain.agents import initialize_agent


# other libraries
import csv
import json
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from unidecode import unidecode
from botocore.exceptions import ClientError
from datetime import datetime, timezone, timedelta

# para bm25
from rank_bm25 import BM25Okapi
import nltk
import unicodedata
import re
import dill as pickle
import pandas as pd
from io import BytesIO

import random

nltk.data.path.append('/usr/share/nltk_data')


"""
Possibles Models

"anthropic": "anthropic.claude-3-sonnet-20240229-v1:0",
"anthropic": "anthropic.claude-v2:1",
"anthropic": "anthropic.claude-v2",
"anthropic" : "anthropic.claude-instant-v1",
"emb-titan-v1":"amazon.titan-embed-text-v1"
"emb-titan-v1":"amazon.titan-e1t-medium"
"""

# Setup Embedding and LLM constants
MODEL_LIST = {"anthropic":"anthropic.claude-3-haiku-20240307-v1:0",
            "emb-titan-v1": "amazon.titan-embed-g1-text-02"
            }

MODEL_ID_EMB = MODEL_LIST['emb-titan-v1']
MODEL_ID_LLM = MODEL_LIST['anthropic']


bedrock_client = boto3.client('bedrock-runtime')

# Declaring functions

def call_lambda_dynamodb_log(request_id, log_count, timestamp_ref, log_type, chat_id, tenant, data):

    print("Starting call_lambda_dynamodb_log")
    function_name = 'sofia-ai-assistant-write-dynamo-log'

    # Parâmetros para a chamada do Lambda
    payload = data.copy()
    
    payload["request_id"] = request_id + f"-{log_count[0]:04}"
    payload["timestamp_ref"] = timestamp_ref
    payload["log_type"] = log_type
    payload["chat_id"] = chat_id
    payload["tenant"] = tenant

    # Configuração do cliente do AWS Lambda
    client = boto3.client('lambda')

    try:
        # Chamada assíncrona da função Lambda
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='Event',  # Invocação assíncrona
            Payload=json.dumps(payload)
        )

        log_count[0] = log_count[0] + 1
        
        # Verifica o código de status da resposta
        status_code = response['StatusCode']
        if status_code == 202:
            msg = 'Success'
            print(msg)
            return msg
        else:
            msg = 'Fail: Status code ' + str(status_code)
            print(msg)
            return msg

    except Exception as e:
        msg = 'Fail: ' + str(e)
        print(msg)
        return msg
    
# get database secrets 
def get_secret(region_name, secret_name):
    print("Starting get_secret.")

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

    print("get_secret ok.")
    return get_secret_value_response['SecretString']


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
'houvesse', 'houvessem', 'houvéssemos', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais', 'mas', 'me', 'mesmo', 'meu', 'meus',
'minha', 'minhas', 'muito', 'na', 'não', 'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num',
'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 'qual', 'quando', 'que', 'quem', 'são',
'se', 'seja', 'sejam', 'sejamos', 'sem', 'ser', 'será', 'serão', 'serei', 'seremos', 'seria', 'seriam', 'seríamos', 'seu',
'seus', 'só', 'somos', 'sou', 'sua', 'suas', 'também',
'te', 'tem', 'tém', 'temos', 'tenha', 'tenham', 'tenhamos', 'tenho', 'terá', 'terão', 'terei', 'teremos', 'teria', 'teriam', 'teríamos',
'teu', 'teus', 'teve', 'tinha', 'tinham', 'tínhamos', 'tive', 'tivemos', 'tiver', 'tivera', 'tiveram', 'tivéramos', 'tiverem',
'tivermos', 'tivesse', 'tivessem', 'tivéssemos', 'tu', 'tua', 'tuas', 'um', 'uma', 'você', 'vocês',
'vos']

stopwords_sm = {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas', 'para', 'com'}
greetings = {'ola', 'oi', 'alô', 'olá', 'hey', 'hello', 'bom dia', 'boa tarde', 'boa noite'}


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


def preprocess_string(txt, remove_stop=True, do_stem=True, to_lower=True):
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
    txt = strip_accents(txt)
    
    if to_lower:
        txt = txt.lower()
    tokens = nltk.tokenize.word_tokenize(txt, language="portuguese")
    
    if remove_stop:
        tokens = [tk for tk in tokens if tk not in stop_words]
    if do_stem:
        stemmer = nltk.stem.PorterStemmer() 
        tokens = [stemmer.stem(tk) for tk in tokens]
    return tokens


def search_pgvector_results(vector_store, query, topk):
    print("Starting search_pgvector_results") 

    vector_store.as_retriever(search_type = "similarity", 
                                        search_kwargs = {"k" : 1}, 
                                        verbose=True,
                                        return_intermediate_steps=True)
        
    res = vector_store.similarity_search(query, k=topk)
    
    print(f"Returned {len(res)} results.")
    
    rank=1
    dict_faq = {}
    for res_ind in range(topk - 1):
        b_ = res[res_ind].page_content
        s_ = res[res_ind].metadata['source']
        t_ = res[res_ind].metadata['title']
        #i_ = res[res_ind].metadata['id']
        
        dict_faq[i_] = {'pergunta': t_, 'resposta': b_, 'link': s_}

        rank+=1

    return dict_faq


def prepare_prompt_and_call(prompt_instruction, question, content_list, parameters):
    print("Starting prepare_prompt_and_call") 
    
    prompt_template = f"""
    \n\nHuman:

    {prompt_instruction}
    
    A pergunta e o contexto estão abaixo.

    Pergunta: {{question}}
    ==========
    {{content}}
    ==========
    
    Assistant:
    """

    #content_string = "\n".join([f"<context>\n<pergunta>{content_dict[item]['pergunta']}</pergunta>\n<resposta>{content_dict[item]['resposta']}</resposta>\n<source>{content_dict[item]['link']}</source></context>" for item in content_dict])
    content_string = "\n".join([f"<context>\n<pergunta>{item['pergunta']}</pergunta>\n<resposta>{item['resposta']}</resposta>\n<source>{item['link']}</source></context>" for item in content_list])
        
    prompt_template = prompt_template.replace('{question}',question)
    
    prompt_template = prompt_template.replace('{content}',content_string)
        
    q_ = f'Responda a seguinte pergunta: {question}'
    
    llm_start = datetime.now()
    r_ = run_evaluatorClaude3(q_, prompt_template, parameters)
    llm_end = datetime.now()

    return r_[0], prompt_template, llm_end - llm_start


def create_langchain_vector_embedding_using_bedrock():
    print("Starting create_langchain_vector_embedding_using_bedrock") 
    bedrock_embeddings_client = BedrockEmbeddings(
        client=bedrock_client,
        model_id=MODEL_ID_EMB)
    return bedrock_embeddings_client


def search_query(bm25, query, k=1,doStopRemoval=True,doStem=True, doLower = True):
    print("Starting search_query") 

    pergunta_ = preprocess_string(query,doStopRemoval,doStem,doLower)

    scores_ = bm25.get_scores(pergunta_)   
    most_relevant_documents_ = np.argsort(-scores_)
    return most_relevant_documents_[:k]


def get_data_bm25(string_resultado, engine):
    print("Starting get_data_bm25") 
    
    Session = sessionmaker(bind=engine)
    session = Session()

    # Nome da tabela que você quer acessar
    nome_tabela = 'sofia_ai_assistant_corpos_bm25'

    # Consultando os dados (use a função text para garantir que a consulta seja tratada como texto)
    consulta_sql = text(f'SELECT * FROM {nome_tabela} where index = {string_resultado}')
    resultados = session.execute(consulta_sql).fetchall()

    # Encerrando a sessão
    session.close()
    return resultados


def run_evaluatorClaude3(pergunta, system_prompt, parameters):
    print("Starting run_evaluatorClaude3") 

    messages = [{ "role":'user', "content":[{'type':'text','text': pergunta}]}]

    body = json.dumps({
        "messages": messages,
        "temperature": parameters["temperature"], 
        "top_p": parameters["top_p"], 
        "top_k": parameters["top_k"], 
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": parameters["max_tokens"],
        "system": system_prompt
        })

    modelId     = MODEL_ID_LLM  
    accept      = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    
    response_body = json.loads(response.get("body").read())
    return response_body.get("content")


def capfirst(s):
    return s[:1].upper() + s[1:]


# TODO: Insert "infelizmente" conditions here?
def cleanAnswer(answer):
    list_remove = ['\s*De acordo com as informações internas consultadas[,:]\s*',
                '\s*com base nas informações internas consultadas,\s*',
                '\s*De acordo com as informações fornecidas[,:]\s*',
                '\s*De acordo com as informações consultadas[,:]\s*',
                '\s*Baseado nas informações internas consultadas[,:]\s*',
                '\s*Baseado nas informações consultadas[,:]\s*',
                '<response>',
                '</response>',
                'O score\s.*\s\d\.\d\.']

    for str_remov in list_remove:
        answer = re.sub(str_remov, '', answer,flags=re.IGNORECASE)
    
    answer = capfirst(answer)
    return answer


def cleanNaoSei(phrase):
    patterns = ['Não sei te responder',
            'Não sei responder',
            'Não sei informar',
            'não consigo responder',
            'não consigo encontrar',
            'Não sei te responder',
            'Não encontrei informações',
            'Não tenho informações']
    
    msg_padrao = 'Não sei te responder.'
    
    if len(phrase)<1:
        return msg_padrao
    
    # Convert the phrase to lowercase for case-insensitive comparison
    phrase_lower = phrase.lower()

    # Check each pattern in the list
    for pattern in patterns:
        # Convert the pattern to lowercase
        pattern_lower = pattern.lower()

        # Check if the pattern is in the phrase
        if re.search(pattern_lower, phrase_lower):
            return msg_padrao

    # If no pattern matches
    return capfirst(phrase.strip())


def cleanReason(phrase):
    re_ = re.search(r'<reason>(.*?)</reason>',phrase,flags=re.DOTALL)
    phrase = re.sub(r'<reason>.*?</reason>',' ',phrase,flags=re.S)
    if re_ != None:
        re_ = re_[1]
    return re_,capfirst(phrase.strip())


def cleanSource(phrase):
    re_ = re.search(r'<source>(.*?)</source>',phrase,flags=re.DOTALL)
    phrase = re.sub(r'<source>.*?</source>',' ',phrase,flags=re.S)
    if re_ != None:
        re_ = re_[1]
    return re_,capfirst(phrase.strip())


def cleanFunctionCalls(phrase):
    phrase = re.sub(r'<function_calls>.*?</function_calls>','',phrase,flags=re.S)
    return capfirst(phrase.strip())
    

def cleanResposta(phrase):
    phrase = re.sub(r'Resposta','',phrase,flags=re.S)
    return capfirst(phrase.strip())


def add_space_before_punctuation(text):
    # Use a regular expression to find any word followed directly by a question mark
    # and replace it by adding a space before the question mark
    updated_text = re.sub(r'(\w)(\?)', r'\1 \2', text)
    return updated_text


# ini is_valid_question #
def is_valid_question(sentence, stopwords, greetings):
    print("Starting is_valid_question") 

    # Rule 1: Check if sentence starts with '/'
    if sentence.startswith('/'):
        return False, 'Rule 1 violated: Sentence starts with "/"'

    # Rule 2: Preliminary check on the length of the original sentence
    tamanho = len(sentence)
    tamanho_w = len(sentence.split())

    if (tamanho >= 120):
        return False, 'Rule 2 violated: Sentence has 120 or more characters.'

    if (tamanho_w <= 2):
        return False, 'Rule 2.1 violated: Sentence has less than 3 words.'

    # Convert sentence to lowercase and remove accents
    processed_sentence = unidecode(sentence.lower())

    # Remove non-alphanumeric characters (except spaces and question marks)
    processed_sentence = re.sub(r"[^\w\s?]", "", processed_sentence)

    processed_sentence = add_space_before_punctuation(processed_sentence)

    # Rule 3: Check for the presence of invalid words
    invalid_words = {'voce', 'voces', 'atendente', 'atendentes','humano', 'vc', 'vcs'}
    #print(processed_sentence)
    if any(word in processed_sentence.split() for word in invalid_words):
        return False, 'Rule 3 violated: Sentence contains invalid words.'

    # Clean sentence from stopwords and greetings
    # Replace compound greetings with a space
    for greeting in greetings:
        processed_sentence = re.sub(r"\b" + greeting + r"\b", " ", processed_sentence)

    words = [word for word in processed_sentence.split() if word not in stopwords]
    significant_words_count = len(words)

    processed_sentence = ' '.join(words)
    tamanho = len(processed_sentence)
    tamanho_w = len(words)

    # Re-check the length after processing
    if (tamanho >= 120):
        return False, 'Rule 4 violated: Sentence has 120 or more characters, after processing'

    if (tamanho_w <= 2):
        return False, 'Rule 4.1 violated: Sentence has less than 2 words, after processing'


    # # Rule 5: Check for question mark or question keywords
    # question_keywords = {'porque', 'por que', 'quero', 'queria', 'poderia','como '}
    # #print(processed_sentence)
    # if '?' not in processed_sentence and not any(keyword in processed_sentence for keyword in question_keywords):
    #     return False, 'Rule 5 violated: No question mark or question keywords'

    return True, ''
    # fim is_valid_question #


def createLink(text):
    print("Starting createLink") 
    # Normalize and remove accents
    normalized_text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

    # Lowercase the string
    lowercased_text = normalized_text.lower()

    # Replace spaces with hyphens and remove punctuation (except hyphens)
    final_text = re.sub(r'[^\w\s-]', '', lowercased_text).replace(' ', '-')
    return final_text


def format_return(answer,source_str,statusCode=200):
    print("Starting format_return") 

    # Removing source link when default answer
    if answer == 'Não sei te responder.' or source_str == '-':
        source_str = ''

    resp_dict = {"resposta": answer,"source_link":source_str} 
    return {
        "isBase64Encoded": "false",
        "statusCode": statusCode,
        "headers": {
            "Access-Control-Allow-Headers" : "Content-Type",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
        },
        "body": json.dumps(resp_dict)
    }
    

def returnNaoSeiStatus(frase):
    if frase == 'Não sei te responder.':
        return 210
    else: 
        return 200


# rewrite_query inicio
def rewrite_query(query):
    # Remove accents
    query = unidecode(query)

    # Define patterns and replacements in a dictionary
    replacements = {
        r'nao presta': 'nao funciona',
        r'nao esta prestando': 'nao esta funcionando'
    }

    # Apply general replacements
    for pattern, replacement in replacements.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)

    # Handle "desbloquear" with dynamic possessive and variations of "maquina"
    # Use a regular expression to match "desbloquear" followed by any one word, and variations of "maquina"
    #query = re.sub(r'desbloque(ar|ie) (\w+) (maquina|maquininha)', r'ativ\1 \2 \3', query, flags=re.IGNORECASE)
    pattern = r'desbloque(ar|ie)( o| a| os| as)? (\w+)? (maquina|maquininha)'
    query = re.sub(pattern, lambda m: f'ativ{"ar" if m.group(1) == "ar" else "e"}{m.group(2) if m.group(2) else ""} {m.group(3) if m.group(3) else ""} {m.group(4)}', query, flags=re.IGNORECASE)

    # consignado
    # Define a pattern to check if 'emprestimo' or its variations are already in the query
    emprestimo_pattern = re.compile(r'\b(empr[eé]stimo|empr[eé]stimos)\b', re.IGNORECASE)

    # Check if 'emprestimo' or its variations are in the query
    if emprestimo_pattern.search(query):
        return query  # Return the original query if it already contains 'emprestimo'

    # Define the pattern to find 'consignado'
    consignado_pattern = re.compile(r'\bconsignado\b', re.IGNORECASE)

    if consignado_pattern.search(query):
        if not emprestimo_pattern.search(query):
            # Replace 'consignado' with 'emprestimo consignado' if not preceded by any form of 'emprestimo'
            query = consignado_pattern.sub('emprestimo consignado', query)

    return query   
    # rewrite_query fim




def search_bm25_results(bm25, bm25_corpus, query, topk):
    # Get bm25 similarity result
    res = search_query(bm25, query, k=topk)

    list_faq = []
    for ind in range(len(res)):
        string_resultado = int(str(res[ind]))

        #ID_FAQ          = bm25_corpus.iloc[string_resultado].qid
        PERGUNTA_FAQ    = bm25_corpus.iloc[string_resultado].pergunta
        RESPOSTA_FAQ    = bm25_corpus.iloc[string_resultado].resposta
        LINK_FAQ        = bm25_corpus.iloc[string_resultado].link

        #cod = extract_url_code(LINK_FAQ)
        list_faq.append({'pergunta': PERGUNTA_FAQ, 'resposta': RESPOSTA_FAQ, 'link': LINK_FAQ})

    return list_faq


def search_pgvector_results(vector_store, query, topk):
    print("Starting search_pgvector_results") 

    vector_store.as_retriever(search_type = "similarity", 
                                        search_kwargs = {"k" : 1}, 
                                        verbose=True,
                                        return_intermediate_steps=True)
        
    res = vector_store.similarity_search(query,k=topk)

    print(f"Returned {len(res)} results.")
    
    rank=1
    list_faq = []
    for res_ind in range(topk - 1):
        b_ = res[res_ind].page_content
        s_ = res[res_ind].metadata['source']
        t_ = res[res_ind].metadata['title']
        #i_ = res[res_ind].metadata['id']
        
        list_faq.append({'pergunta': t_, 'resposta': b_, 'link': s_})

        rank+=1

    return list_faq


def handler(event, context):
    """
    Lambda handler function, will be executed when the lambda is called.

    Here, we expect a body (event['body']) with the following parameters:
    body['pergunta']: Question being sent to the LLM after processing
    body['id_chat']:  ID of the chat configuration
    """ 

    session_ = round(random.random()*10000000000000)
    print(f"Starting session [{session_}]")
    print("event", event)
    print("event_type", type(event))
    print("event_keys", event.keys())

    call_start = datetime.now()

    # Using a list so its mutable
    log_count = [1]
    
    # Getting environment variables. Those are set by the lambda configuration.
    secret_name = os.environ.get('SECRET_NAME')
    region_name = os.environ.get('REGION_NAME')
    pgvector_host = os.environ.get('PGVECTOR_HOST')
    s3_artifacts_bucket = os.environ.get('S3_ARTIFACTS_BUCKET')
    s3_data_bucket = os.environ.get('S3_DATA_BUCKET')

    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    boto3_bedrock = boto3.client('bedrock-runtime')
    

    # Getting secrets data for authentication
    secrets_db = json.loads(get_secret(region_name, secret_name))
    user_db_ = secrets_db["username"]
    print(user_db_)
    password_db_ = secrets_db["password"]


    # Setting database configurations. Those will be used to retrieve the PGVector and BM25 contexts
    PGVECTOR_DRIVER         = "psycopg2"
    PGVECTOR_USER           = user_db_
    PGVECTOR_PASSWORD       = password_db_
    PGVECTOR_HOST           = pgvector_host
    PGVECTOR_PORT           = 5432
    PGVECTOR_DATABASE       = "postgres"
    

    # Get input data
    tenant = ''
    id_chat = 0
    try:
        body = json.loads(event['body'])
        pergunta_original = body['pergunta']
        tenant = body['tenant']
        id_chat = int(body['id_chat'])
    except KeyError as err:
        msg = 'Parâmetros inválidos: ' + str(err)
        print(msg)
        call_end = datetime.now()
        call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "return_invalid_parameters", id_chat, tenant,
                                            {"full_duration": str(call_end - call_start)})
        return format_return(msg,'',statusCode=400)
    except Exception as err:
        msg = 'Erro ao carregar dados de entrada: ' + str(err)
        print(msg)
        call_end = datetime.now()
        call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "return_invalid_parameters", id_chat, tenant,
                                            {"full_duration": str(call_end - call_start)})
        return format_return(msg,'',statusCode=400)


    # Preprocessing the question
    rewritten_question = rewrite_query(pergunta_original)
    
    # Question validation
    validaSentenca = is_valid_question(rewritten_question, stopwords_sm, greetings)

    print(f'Session:{session_} Original question: {pergunta_original}')
    print(f'Session:{session_} ValidaSentenca: {validaSentenca[0]}, {validaSentenca[1]}')
    if not validaSentenca[0]:
        answer = "Peço desculpas, não compreendi a sua pergunta. Poderia reformulá-la com outras palavras?"
        call_end = datetime.now()
        # If the question is invalid, execute dynamo logging and return error
        call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "return_invalidation", id_chat, tenant,
                                 {"full_duration": str(call_end - call_start), 
                                  "original_question":pergunta_original,
                                  "final_question": rewritten_question,
                                  "invalidation_rule": validaSentenca[1],
                                  "final_answer": answer})
        
        return format_return(answer,'',statusCode=210)
    

    # Bedrock initialization for embeddings
    bedrock_embeddings_client = create_langchain_vector_embedding_using_bedrock()    


    # Preparing the PGVector object to retrieve similar contexts
    CONNECTION_STRING = PGVector.connection_string_from_db_params(                                                  
                driver=PGVECTOR_DRIVER,
                user=PGVECTOR_USER,
                password=PGVECTOR_PASSWORD,
                host=PGVECTOR_HOST,
                port=PGVECTOR_PORT,
                database=PGVECTOR_DATABASE,
            )
            
    print("Preparing PGVector object") 
    docsearch = PGVector(
        connection_string=CONNECTION_STRING, 
        embedding_function=bedrock_embeddings_client, 
        collection_name=f"{id_chat}",
        distance_strategy=DistanceStrategy.COSINE
    )


    # Reading files from S3
    s3 = boto3.client('s3')
    model_version = 1

    try:
        # Reading bm25 pickle and corpus files
        pkl_object_key = f"{tenant}/{id_chat}/{model_version}/bm25.pickle"
        pkl_obj = s3.get_object(Bucket=s3_artifacts_bucket, Key=pkl_object_key)
        with BytesIO(pkl_obj['Body'].read()) as handle:
            bm25 = pickle.load(handle)    

        corpus_object_key = f"{tenant}/{id_chat}/data/chat_data.csv"
        corpus_obj = s3.get_object(Bucket=s3_data_bucket, Key=corpus_object_key)
        bm25_corpus = pd.read_csv(corpus_obj['Body'])

        # Reading prompt file
        prompt_object_key = f"{tenant}/{id_chat}/prompt/prompt.txt"
        prompt_obj = s3.get_object(Bucket=s3_data_bucket, Key=prompt_object_key)
        prompt_instruction = prompt_obj['Body'].read().decode("utf-8")
    
    except Exception as exc:
        msg = 'Erro ao obter arquivos: ' + str(exc)
        print(msg)
        call_end = datetime.now()
        call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "return_server_error", id_chat, tenant,
                                            {"full_duration": str(call_end - call_start)})
        return format_return(msg,'',statusCode=500)

    # Reading parameters file
    # parameters_object_key = f"{tenant}/{id_chat}/parameters/parameters.txt"
    # parameters_obj = s3.get_object(Bucket=s3_data_bucket, Key=parameters_object_key)
    # parameters = json.loads(parameters_obj['Body'].read().decode("utf-8"))

    # Temporary static parameters (TODO: Use file when frontend start writing it)
    parameters = json.loads("""{
        "temperature": 0.01, 
        "top_p": 0.5, 
        "top_k": 1,
        "max_tokens": 512
    }""")




    # Retrieve BM25 similar results
    content_bm25 = search_bm25_results(bm25, bm25_corpus, rewritten_question, topk=1)

    # Retrieve PGVector similar results
    content_pgvector = search_pgvector_results(docsearch, rewritten_question, topk=4)

    # Merge contents
    #merged_content = {**content_bm25, **content_pgvector}
    merged_content = content_bm25 + content_pgvector

    # If there is no relevant context, return error
    if len(merged_content) == 0:
        msg = 'Nenhum conteúdo relevante encontrado.'
        print(msg)
        call_end = datetime.now()
        call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "return_server_error", id_chat, tenant,
                                            {"full_duration": str(call_end - call_start)})
        return format_return(msg,'',statusCode=500)


    # Prepare final prompt and call LLM
    answer_,final_prompt, llm_duration = prepare_prompt_and_call(prompt_instruction, rewritten_question, merged_content, parameters)


    # Process LLM response (TODO: is it necessary?)
    answer = answer_['text']
    answer2 = re.sub(r'\r\n?|\n', '|||', answer)
    print(f'Session:{session_} Original answer: {answer2}')

    source_str,answer = cleanSource(answer) 
    reason_str,answer = cleanReason(answer)
    clean_answer = cleanFunctionCalls(cleanNaoSei(cleanAnswer(answer)))

    # Dynamo Logging
    call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "llm_answer", id_chat, tenant,
                                {"final_question": rewritten_question,
                                    "original_answer": answer,
                                    "source_link": source_str,
                                    "final_prompt": final_prompt,
                                    "llm_answer_duration": str(llm_duration)})
    
    clean_answer2 = re.sub(r'\r\n?|\n', '|||', clean_answer)
    print(f'Session:{session_} Final answer: {clean_answer2}')
    print(f'Session:{session_} Source link: {source_str}')

    # Dynamo Logging
    call_end = datetime.now()
    call_lambda_dynamodb_log(context.aws_request_id, log_count, int(datetime.now(timezone.utc).timestamp()), "return_answer", id_chat, tenant,
                                 {"full_duration": str(call_end - call_start), 
                                  "original_question": pergunta_original,
                                  "final_question": rewritten_question, 
                                  "original_answer": answer, 
                                  "final_answer": clean_answer, 
                                  "source_link": source_str})
    
    return format_return(clean_answer, source_str, statusCode=returnNaoSeiStatus(clean_answer))

