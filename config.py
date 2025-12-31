import os
from pathlib import Path
from dotenv import load_dotenv


env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "contentguard")
    RAY_ADDRESS = os.getenv("RAY_ADDRESS", "auto")

config = Config()