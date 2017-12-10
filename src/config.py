import os.path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.abspath(os.path.relpath("../res", ROOT_DIR))
DATA_DIR = os.path.join(RES_DIR, "data")
