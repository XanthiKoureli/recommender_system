import os
from langchain_mistralai import ChatMistralAI
from config.setting import settings
from langchain_openai import ChatOpenAI
# os.environ["MISTRAL_API_KEY"] = settings.MISTRAL_API_KEY

# llm_mistral = ChatMistralAI(
#     model="mistral-large-latest",
#     temperature=0,
#     max_retries=2,
# )

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
)
