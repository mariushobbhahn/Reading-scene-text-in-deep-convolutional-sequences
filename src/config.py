import os.path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.abspath(os.path.relpath("../../../res", ROOT_DIR))


__SERVER_DATA = os.path.abspath("/graphics/projects/scratch/student_datasets/cgpraktikum17/deep-sequences/")
__LOCAL_DATA = os.path.join(RES_DIR, "data")

IS_SERVER = os.path.exists(__SERVER_DATA)


DATA_DIR = __SERVER_DATA if IS_SERVER else __LOCAL_DATA

