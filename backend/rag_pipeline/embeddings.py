from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from config.setting import settings
# https://github.com/langchain-ai/langchain/issues/20618

if settings.MODEL_NAME == 'openai':
    embeddings_function = OpenAIEmbeddings(openai_api_key=settings.OPEN_API_KEY,
                                           model="text-embedding-3-large",)
else:
    embeddings_function = MistralAIEmbeddings(api_key=settings.MISTRAL_API_KEY)

