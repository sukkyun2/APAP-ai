from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    history_api: str = Field(alias='HISTORY_API_HOST', default='http://127.0.0.1:8080')
    yolo_weight_path: str = Field(default='yolov5su.pt')

    class Config:
        env_file = '.env'


settings = Settings()
