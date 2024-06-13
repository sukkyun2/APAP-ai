from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    history_api: str = Field(alias='history_api_host', default='localhost:8080')

    class Config:
        env_file = '.env'


settings = Settings()
