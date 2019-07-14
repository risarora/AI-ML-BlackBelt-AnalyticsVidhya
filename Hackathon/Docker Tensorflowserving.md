# Docker Setup

## Install Keras from PyPI (recommended):
sudo pip install keras
If you are using a virtualenv, you may want to avoid using sudo:

pip install keras

touch $HOME/.keras/keras.json


## Setup Keras Backend

touch $HOME/.keras/keras.json

```
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "tensorflow"
}

```


## Docker Command
```
docker pull tensorflow/serving
docker run -p 8501:8501 --name tfserving_resnet \
--mount type=bind,source=/tmp/resnet,target=/models/resnet \
-e MODEL_NAME=resnet -t tensorflow/serving &


```

http://localhost:8501/v1/models/resnet
