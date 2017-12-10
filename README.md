# CGPraktikum
Repositories for the course **Computer Graphics I** at the Eberhard Karls University Tübingen.

## Setup

(Does not fully work yet)

```bash
ssh username@cgpool120[0-7].informatik.uni-tuebingen.de # (or cgcontact first if not in wsi network)

pip install virtualenv --user
pip install virtualenvwrapper --user

nano ~/.profile
```

Add the following lines to the file:

```
export PATH="/graphics/projects/cuda/toolkit_deeplearning/cuda/bin:$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="/graphics/projects/cuda/toolkit_deeplearning/cuda/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="$HOME/.local/lib/python2.7/site-packages:$PYTHONPATH"
export WORKON_HOME=$HOME/.virtualenvs
export PROJECT_HOME=$HOME/PCG
source $HOME/.local/bin/virtualenvwrapper.sh
export CUDA_VISIBLE_DEVICES='0'
```

(note the addition of PYTHONPATH compared to the tutorial in ILIAS)  
Press `Ctrl+O`, `Enter` and `Ctrl+X` to save/exit the profile

Then (everytime you log on), type:

```bash
source ~/.profile
mkvirtualenv env
```

to start the environment.  
The first time in the environment, install tensorflow:

```
easy_install -U pip        #ensure pip version
pip install --user --upgrade tensorflow-gpu
pip install --user -U git+https://github.com/ppwwyyxx/tensorpack.git
```

This should install tensorflow aswell as tensorpack (tensorpack didn't work for me without adding site-packages directory to $PYTHONPATH).

Test if it is correctly installed using

```python
import tensorflow as tf
from tensorpack import *
```

If this doesn't fail, the packages should be installed correctly

- tensorflow getting started: https://www.tensorflow.org/get_started/get_started
