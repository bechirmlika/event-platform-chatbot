from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    MYSQL_HOST: str
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_DATABASE: str
    MYSQL_PORT: int = 3306  # optional with default

    GEMINI_API_KEY: str

    PINECONE_API_KEY: str
    PINECONE_INDEX: str = "evey-db"  # default
    PINECONE_CLOUD: str = "aws"      # default
    PINECONE_REGION: str = "us-east-1"  # default

    # This tells pydantic-settings to load from `.env` file if it exists
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
