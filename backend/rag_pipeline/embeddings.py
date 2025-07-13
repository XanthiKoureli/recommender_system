from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config.setting import settings
# https://github.com/langchain-ai/langchain/issues/20618
from backend.rag_pipeline.pubmed_bert_embedder import PubMedBERTEmbeddings
from backend.rag_pipeline.sap_embedder import SapBERTEmbeddings


if settings.MODEL_NAME == 'openai':
    embeddings_function = OpenAIEmbeddings(openai_api_key=settings.OPEN_API_KEY,
                                           model="text-embedding-3-large",)
elif settings.MODEL_NAME == 'mistral':
    embeddings_function = MistralAIEmbeddings(api_key=settings.MISTRAL_API_KEY)
    
elif settings.MODEL_NAME == 'pubmedbert':
    embeddings_function = PubMedBERTEmbeddings()
    
elif settings.MODEL_NAME == 'sapbert':
    embeddings_function = SapBERTEmbeddings()
else:
    raise ValueError(f"Unsupported MODEL_NAME: {settings.MODEL_NAME}")

