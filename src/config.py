import os.path
import getpass


ROOT_DIR = os.path.abspath(__file__)
ROOT_DIR_NAME = os.path.dirname(ROOT_DIR)
RES_DIR = os.path.abspath(os.path.relpath("../res", os.path.abspath(__file__)))


__SERVER_DATA = os.path.abspath("/graphics/projects/scratch/student_datasets/cgpraktikum17/deep-sequences/")
__LOCAL_DATA = os.path.join(RES_DIR, "data")

IS_SERVER = os.path.exists(__SERVER_DATA)


DATA_DIR = __SERVER_DATA if IS_SERVER else __LOCAL_DATA

TRAIN_LOG_DIR_NAME = "train_log/{}".format(getpass.getuser())
TRAIN_LOG_DIR = os.path.join(__SERVER_DATA if IS_SERVER else ROOT_DIR, TRAIN_LOG_DIR_NAME)

DUMP_DIR = None
REMOVE_LMDB = False
