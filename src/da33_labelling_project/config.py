#%% LIB
from pathlib import Path

#%% DIR
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DATA_DIR =  DATA_DIR / 'processed'
RAW_DATA_DIR = DATA_DIR / 'raw'

