from pydantic import BaseSettings


class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_DEFAULT_REGION: str

    class Config:
        env_file = "./.env"


settings = Settings()
