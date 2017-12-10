import config
import os
import cv2

from data.dataset import lmdb_path
from tensorpack.dataflow import *
from data.iiit5k import IIIT5K

# Use this values to toggle between train and test data
MODE = 'train'
REUSE_LMDB = True


path = lmdb_path(IIIT5K, MODE)

# Check fi the old lmdb file should be removed
if not REUSE_LMDB:
    os.remove(path)

# Check if data set is already stored as lmdb
if not os.path.exists(path):
    # If not, read data set and dump it to lmdb
    tmp_data = IIIT5K('train', shuffle=False)
    tmp_data.dump_to_lmdb(config.DATA_DIR)


#load data
ds = LMDBData(path, shuffle=False)
ds = LocallyShuffleData(ds, 50000)
ds = LMDBDataPoint(ds)
ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)

ds.reset_state()

#Print labels of data
for (img_str, label) in ds.get_data():
    print(label)




