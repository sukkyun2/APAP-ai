from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    history_api: str = Field(alias='HISTORY_API_HOST', default='http://localhost:8080')
    yolo_weight_path: str = Field(alias='YOLO_WEIGHT_PATH', default='yolov5su.pt')
    custom_yolo_weight_path: str = Field(alias='CUSTOM_YOLO_WEIGHT_PATH', default='yolov5su.pt')
    allow_origins: str = Field(alias='ALLOW_ORIGINS', default='')  # TODO Convert to List

    class Config:
        env_file = '.env'


settings = Settings()
