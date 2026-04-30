#%% LIB
from dotenv import load_dotenv
from openai import OpenAI
import os
import json

from .config import PROJECT_ROOT

#%% LLM MODELS
load_dotenv(PROJECT_ROOT / ".env")

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    return client

#%% JSON
def load_records(data_dir) -> list[dict]:
    files = [file for file in data_dir.glob("*.json")]
    data_holder = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
            data_holder.extend(temp_data)
    return data_holder

def save_records(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    return {
        'message': f'Save data to {path}',
        'status': 'success'
        }