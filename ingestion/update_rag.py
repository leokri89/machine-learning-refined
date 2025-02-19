secret_name = os.environ.get('SECRET_NAME')
region_name = os.environ.get('REGION_NAME')
pgvector_host = os.environ.get('PGVECTOR_HOST')
s3_bucket_name_dados = os.environ.get('S3_BUCKET_NAME_DADOS')
s3_bucket_name_artefatos = os.environ.get('S3_BUCKET_NAME_ARTEFATOS')
dynamo_table_name =os.environ.get('DYNAMO_TABLE_NAME')

tenant_id = 1
chat_id = 1

MODEL_ID_EMB = "amazon.titan-embed-g1-text-02"

PGVECTOR_DRIVER = "psycopg2"
PGVECTOR_USER = "user"
PGVECTOR_PASSWORD = "password"
PGVECTOR_HOST = pgvector_host
PGVECTOR_PORT = 5432
PGVECTOR_DATABASE = "postgres"
COLLECTION_NAME = chat_id
DISTANCE_STRATEGY = "DistanceStrategy.EUCLIDEAN"

os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

bedrock_client = boto3.client('bedrock-runtime')

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

stemmer = nltk.stem.PorterStemmer()

nlp = spacy.load('en_core_web_sm')

embeddings_client = BedrockEmbeddings(client=bedrock_client, model_id=MODEL_ID_EMB)

BASE_DADOS=f"{tenant_id}/{chat_id}/data/chat_data.csv"

text_chunksize=2500
text_chunkover=10t
text_splitter = SpacyTextSplitter(chunk_size = text_chunksize, chunk_overlap = (int(text_chunkover)/100)*text_chunksize )

#docs = read_csv_file_spacy_chunk_s3(s3_bucket_name_dados, BASE_DADOS, text_splitter, delim=',')

s3 = boto3.client('s3')

s3_object = s3.get_object(Bucket=s3_bucket_name_dados, Key=BASE_DADOS)

file_content = s3_object['Body'].read().decode('utf-8')

file = StringIO(file_content)

csv_reader = csv.DictReader(file, delimiter = ',')

descricao_list = []

count = 0
for row in csv_reader:
    count += 1
    docs = text_splitter.split_text(row['resposta'])
    for d_ in docs:
        doc = Document(page_content=row['pergunta'] + ' ' + d_, metadata={'source': row['link'], 'title': row['pergunta'], 'id': row['qid']})
        descricao_list.append(doc)

docs = descricao_list

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
    distance_strategy=DISTANCE_STRATEGY
)

db = PGVector.from_documents(
    documents=docs,
    embedding=bedrock_embeddings_client,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    pre_delete_collection=True,
)

s3_client = boto3.client('s3')
model_version = 1
object_key_bm25 = f"{tenant_id}/{chat_id}/{model_version}/bm25.pickle"
corpus_object_key = BASE_DADOS

corpus_obj = s3_client.get_object(Bucket=s3_bucket_name_dados, Key=corpus_object_key)
corpus_ = pd.read_csv(corpus_obj['Body'], delimiter=',')

corpus_list_ = list(corpus_['resposta'])
#tokenized_corpus = [preprocess_string(doc,True,True) for doc in corpus_list_]

tokenized_corpus = []

remove_stop=True
do_stem=True
to_lower=True
do_lemma = False

for doc in corpus_list_:
    if to_lower:
        doc = doc.lower()

    if do_lemma:
        #txt = lemmatize_sentence(txt)
        sentence = nlp(doc)
        lemmatized_sentence = " ".join([token.lemma_ for token in sentence])

    #doc = strip_accents(doc)
    nfkd = unicodedata.normalize('NFKD', doc)
    new_doc = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    doc = re.sub('[^a-zA-Z0-9 \\\']', ' ', new_doc)
            
    tokens = nltk.tokenize.word_tokenize(doc, language="portuguese")

    if remove_stop:
        tokens = [tk for tk in tokens if tk not in stop_words]

    if do_stem:
        tokens = [stemmer.stem(tk) for tk in tokens]

    tokenized_corpus.append(tokens)

bm25 = BM25Okapi(tokenized_corpus, k1=0.1, b=0.8)

with BytesIO() as data:
    pickle.dump(bm25, data)
    data.seek(0) 

    s3_client.upload_fileobj(data, s3_bucket_name_artefatos, object_key_bm25)

dynamodb = boto3.client('dynamodb')

utc_minus_3 = timezone(timedelta(hours=-3))

response = dynamodb.put_item(
    TableName = dynamo_table_name,
    Item = {
        'chat_id': {
            'S': id_chat
        },
        'tenant': {
            'S': tenant_id
        },
        'status': {
            'S': 'inservice'
        },
        'statusTimestamp': {
            'S': datetime.now(utc_minus_3).strftime('%Y-%m-%d %H:%M:%S')
        },
        'status_message':{
            'S': 'chat created successfully'
        }
    }
)