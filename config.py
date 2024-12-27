import os
from langchain_openai import AzureChatOpenAI
from langchain_community.embeddings import JinaEmbeddings
from langchain_groq import ChatGroq

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_groq import ChatGroq

# load_dotenv()
# aryn_api_key = os.getenv('aryn_api_key')
# jina_api_key = os.getenv('jina_api_key')
# groq_api_key = os.getenv('groq_api_key')

manual_path = 'technical_manual.pdf'     ## solution manual pdf path
summary_filename = 'table_summaries.json'  ## generated summaries of the tables present in the pdf
vector_store_path = "./fox_base_task"       ## persistant directory for vector store      #"./fox_base_task" 
Embedding_model=  'jina-embeddings-v3'     ## Name of the open source embedding model used
llm_name = "llama-3.3-70b-versatile"       ## Name of the open source LLM used
k_value = 15                               ## Number of chunks to retrieve

jina_api_key = 'set here'
groq_api_key = 'set here'
aryn_api_key = 'set here'


llm = ChatGroq(
    temperature=0, 
    groq_api_key=groq_api_key, 
    model_name=llm_name
)

text_embeddings = JinaEmbeddings(
    jina_api_key=jina_api_key, model_name=Embedding_model
)
