import os

from pydantic_settings import BaseSettings, SettingsConfigDict

# Find .env file in the current or parent directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Parent directory
ENV_PATH = os.path.join(BASE_DIR, ".env")


class Settings(BaseSettings):
    # paths
    PROJECT_DIR: str = BASE_DIR
    DATA_DIR: str = os.path.join(PROJECT_DIR, "data")
    MODEL_DIR: str = os.path.join(PROJECT_DIR, "models")
    LOG_DIR: str = os.path.join(PROJECT_DIR, "logs")

    # env variables settings
    model_config = SettingsConfigDict(
        env_file=ENV_PATH,  # multiple files env_file=(".env", ".env.prod")
        env_file_encoding="utf-8",
        extra="allow",
        case_sensitive=True,
    )


# Load settings
settings = Settings()

if __name__ == "__main__":
    # Access values
    print(f"settings : {settings.model_dump()}")
