import os


MAIN_DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data") + os.path.sep
JSON_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", 'json') + os.path.sep
CSV_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "data", 'csv') + os.path.sep
PICS_DIR = os.path.join(MAIN_DATA_DIR, "images") + os.path.sep

os.makedirs(MAIN_DATA_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(PICS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
