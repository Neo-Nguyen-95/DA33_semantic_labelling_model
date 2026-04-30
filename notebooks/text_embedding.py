#%% LIB
import json
# import os
# from pathlib import Path
# from dotenv import load_dotenv

from da33_labelling_project.config import (
    # PROJECT_ROOT, 
    PROCESSED_DATA_DIR
    )

#%% CONFIG
# load_dotenv(PROJECT_ROOT / ".env")
knowledge_data_dir = PROCESSED_DATA_DIR / 'high_school_knowledge'

#%% HELPERS

def load_records(data_dir) -> list[dict]:
    files = [file for file in data_dir.glob("*.json")]
    data_holder = []
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            temp_data = json.load(f)
            data_holder.extend(temp_data)
    return data_holder


#%% MAIN

knowledge_data = load_records(knowledge_data_dir)
knowledge_data[:2]

