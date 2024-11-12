import os
from pydantic.v1 import BaseSettings

# env_path = Path(__file__).parent.parent.parent / '.env'
# config = Config(env_path)


class Settings(BaseSettings):
    OPEN_API_KEY: str = os.getenv("OPENAI_API_KEY")
    MODEL_NAME: str = os.getenv("MODEL_NAME")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY")
    NCBI_API_KEY: str = os.getenv("NCBI_API_KEY")


settings = Settings()
