## Serving ML Quickly with TensorFlow Serving and Docker
Go to the profile of TensorFlow
TensorFlow
Nov 2, 2018
Posted by Gautam Vasudevan, Technical Program Manager, and Abhijit Karmarkar, Software Engineer, Google Brain team

Serving machine learning models quickly and easily is one of the key challenges when moving from experimentation into production. Serving machine learning models is the process of taking a trained model and making it available to serve prediction requests. When serving in production, you want to make sure your environment is reproducible, enforces isolation, and is secure. To this end, one of the easiest ways to serve machine learning models is by using TensorFlow Serving with Docker. Docker is a tool that packages software into units called containers that include everything needed to run the software.


TensorFlow Serving running in a Docker container
Since the release of TensorFlow Serving 1.8, we’ve been improving our support for Docker. We now provide Docker images for serving and development for both CPU and GPU models. To get a sense of how easy it is to deploy a model using TensorFlow Serving, let’s try putting the ResNet model into production. This model is trained on the ImageNet dataset and takes a JPEG image as input and returns the classification category of the image.

Our example will assume you’re running Linux, but it should work with little to no modification on macOS or Windows as well.

Serving ResNet with TensorFlow Serving and Docker
The first step is to install Docker CE. This will provide you all the tools you need to run and manage Docker containers.

TensorFlow Serving uses the SavedModel format for its ML models. A SavedModel is a language-neutral, recoverable, hermetic serialization format that enables higher-level systems and tools to produce, consume, and transform TensorFlow models. There are several ways to export a SavedModel (including from Keras). For this exercise, we will simply download a pre-trained ResNet SavedModel:
```
mkdir /tmp/resnet
curl -s https://storage.googleapis.com/download.tensorflow.org/models/official/20181001_resnet/savedmodels/resnet_v2_fp32_savedmodel_NHWC_jpg.tar.gz | tar --strip-components=2 -C /tmp/resnet -xvz
```
We should now have a folder inside /tmp/resnet that has our model. We can verify this by running:
```
ls /tmp/resnet
1538687457
```
Now that we have our model, serving it with Docker is as easy as pulling the latest released TensorFlow Serving serving environment image, and pointing it to the model:
```
docker pull tensorflow/serving
docker run -p 8501:8501 --name tfserving_resnet \
--mount type=bind,source=/tmp/resnet,target=/models/resnet \
-e MODEL_NAME=resnet -t tensorflow/serving &
…
… main.cc:327] Running ModelServer at 0.0.0.0:8500…
… main.cc:337] Exporting HTTP/REST API at:localhost:8501 …
```
## Breaking down the command line arguments, we are:

* <mark>-p 8501:8501 </mark>: Publishing the container’s port 8501 (where TF Serving responds to REST API requests) to the host’s port 8501
* <mark>--name tfserving_resnet </mark>: Giving the container we are creating the name “tfserving_resnet” so we can refer to it later
* <mark>--mount type=bind,source=/tmp/resnet,target=/models/resnet </mark>: Mounting the host’s local directory (/tmp/resnet) on the container (/models/resnet) so TF Serving can read the model from inside the container.
* <mark>-e MODEL_NAME=resnet </mark>: Telling TensorFlow Serving to load the model named “resnet”
* <mark>-t tensorflow/serving </mark>: Running a Docker container based on the serving image “tensorflow/serving”
Next, let’s download the python client script, which will send the served model images and get back predictions. We will also measure server response times.
```
$ curl -o /tmp/resnet/resnet_client.py https://raw.githubusercontent.com/tensorflow/serving/master/tensorflow_serving/example/resnet_client.py
```
This script will download an image of a cat and send it to the server repeatedly while measuring response times, as seen in the main loop of the script:

```python

# The server URL specifies the endpoint of your server running the ResNet
# model with the name "resnet" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/resnet:predict'

...

# Send few actual requests and time average latency.
total_time = 0
num_requests = 10
for _ in xrange(num_requests):
    response = requests.post(SERVER_URL, data=predict_request)
response.raise_for_status()
total_time += response.elapsed.total_seconds()
prediction = response.json()['predictions'][0]

print('Prediction class: {}, avg latency: {} ms'.format(
prediction['classes'], (total_time*1000)/num_requests))

```

This script uses the requests module, so you’ll need to install it if you haven’t already. By running this script, you should see output that looks like:
```
$ python3 /tmp/resnet/resnet_client.py
Prediction class: 282, avg latency: 185.644 ms

```
As you can see, bringing up a model using TensorFlow Serving and Docker is pretty straight forward. You can even create your own custom Docker image that has your model embedded, for even easier deployment.

## Improving performance by building an optimized serving binary
Now that we have a model being served in Docker, you may have noticed a log message from TensorFlow Serving that looks like:

```
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
```
The published Docker images for TensorFlow Serving are intended to work on as many CPU architectures as possible, and so some optimizations are left out to maximize compatibility. If you don’t see this message, your binary is likely already optimized for your CPU.

Depending on the operations your model performs, these optimizations may have a significant effect on your serving performance. Thankfully, putting together your own optimized serving image is straightforward.

First, we’ll want to build an optimized version of TensorFlow Serving. The easiest way to do this is to build the official Tensorflow Serving development environment Docker image. This has the nice property of automatically generating an optimized TensorFlow Serving binary for the system the image is building on. To distinguish our created images from the official images, we’ll be prepending $USER/ to the image names. Let’s call this development image we’re building $USER/tensorflow-serving-devel:
```
$ docker build -t $USER/tensorflow-serving-devel \
-f Dockerfile.devel \
https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker
```
Building the TensorFlow Serving development image may take a while, depending on the speed of your machine. Once it’s done, let’s build a new serving image with our optimized binary and call it $USER/tensorflow-serving:
```
$ docker build -t $USER/tensorflow-serving \
--build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel \ https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker
```
Now that we have our new serving image, let’s start the server again:
```
$ docker kill tfserving_resnet
$ docker run -p 8501:8501 --name tfserving_resnet \
  --mount type=bind,source=/tmp/resnet,target=/models/resnet \
  -e MODEL_NAME=resnet -t $USER/tensorflow-serving &
```
And finally run our client:
```
$ python /tmp/resnet/resnet_client.py
Prediction class: 282, avg latency: 84.8849 ms
```
On our machine, we saw a speedup of over 100ms (119%) on average per prediction with our native optimized binary. Depending on your machine (and model), you may see different results.

Finally, feel free to kill the TensorFlow Serving container:

```
$ docker kill tfserving_resnet
```
Now that you have TensorFlow Serving running with Docker, you can deploy your machine learning models in containers easily while maximizing ease of deployment and performance.


https://medium.com/tensorflow/serving-ml-quickly-with-tensorflow-serving-and-docker-7df7094aa008
