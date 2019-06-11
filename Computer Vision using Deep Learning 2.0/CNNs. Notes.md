# Computer Vision  Notes

## Intial Setup

* **git package download**

```
git clone https://github.com/risarora/CV101.git
Username: your_username
Password: your_token

```

### Install coda

conda install -c anaconda tensorflow-gpu

```
conda install -c anaconda tensorflow-gpu

crestle@cju6sy2ow001b0130m6d4e1th-87965598c-g4cnr:~$ conda install -c anaconda tensorflow-gpu

Collecting package metadata: done
Solving environment: done


==> WARNING: A newer version of conda exists. <==
  current version: 4.6.8
  latest version: 4.6.14

Please update conda by running

    $ conda update -n base conda



## Package Plan ##

  environment location: /opt/conda

  added / updated specs:
    - tensorflow-gpu


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    _tflow_select-2.1.0        |              gpu           2 KB  anaconda
    absl-py-0.7.1              |           py36_0         157 KB  anaconda
    astor-0.7.1                |           py36_0          43 KB  anaconda
    c-ares-1.15.0              |       h7b6447c_1          98 KB  anaconda
    ca-certificates-2019.5.15  |                0         133 KB  anaconda
    certifi-2019.3.9           |           py36_0         155 KB  anaconda
    cupti-9.0.176              |                0         1.6 MB  anaconda
    gast-0.2.2                 |           py36_0         138 KB  anaconda
    grpcio-1.16.1              |   py36hf8bcb03_1         1.1 MB  anaconda
    keras-applications-1.0.8   |             py_0          33 KB  anaconda
    keras-preprocessing-1.1.0  |             py_1          36 KB  anaconda
    markdown-3.1.1             |           py36_0         113 KB  anaconda
    mock-3.0.5                 |           py36_0          47 KB  anaconda
    openssl-1.1.1              |       h7b6447c_0         5.0 MB  anaconda
    tensorboard-1.13.1         |   py36hf484d3e_0         3.3 MB  anaconda
    tensorflow-1.13.1          |gpu_py36h26cf82e_0           4 KB  anaconda
    tensorflow-base-1.13.1     |gpu_py36h8f37b9b_0       271.6 MB  anaconda
    tensorflow-estimator-1.13.0|             py_0         205 KB  anaconda
    tensorflow-gpu-1.13.1      |       h0d30ee6_0           2 KB  anaconda
    termcolor-1.1.0            |           py36_1           7 KB  anaconda
    werkzeug-0.15.4            |             py_0         262 KB  anaconda
    ------------------------------------------------------------
                                           Total:       284.0 MB

The following NEW packages will be INSTALLED:

  _tflow_select      anaconda/linux-64::_tflow_select-2.1.0-gpu
  absl-py            anaconda/linux-64::absl-py-0.7.1-py36_0
  astor              anaconda/linux-64::astor-0.7.1-py36_0
  c-ares             anaconda/linux-64::c-ares-1.15.0-h7b6447c_1
  cupti              anaconda/linux-64::cupti-9.0.176-0
  gast               anaconda/linux-64::gast-0.2.2-py36_0
  grpcio             anaconda/linux-64::grpcio-1.16.1-py36hf8bcb03_1
  keras-applications anaconda/noarch::keras-applications-1.0.8-py_0
  keras-preprocessi~ anaconda/noarch::keras-preprocessing-1.1.0-py_1
  markdown           anaconda/linux-64::markdown-3.1.1-py36_0
  mock               anaconda/linux-64::mock-3.0.5-py36_0
  tensorboard        anaconda/linux-64::tensorboard-1.13.1-py36hf484d3e_0
  tensorflow         anaconda/linux-64::tensorflow-1.13.1-gpu_py36h26cf82e_0
  tensorflow-base    anaconda/linux-64::tensorflow-base-1.13.1-gpu_py36h8f37b9b_0
  tensorflow-estima~ anaconda/noarch::tensorflow-estimator-1.13.0-py_0
  tensorflow-gpu     anaconda/linux-64::tensorflow-gpu-1.13.1-h0d30ee6_0
  termcolor          anaconda/linux-64::termcolor-1.1.0-py36_1
  werkzeug           anaconda/noarch::werkzeug-0.15.4-py_0

The following packages will be UPDATED:

  ca-certificates    conda-forge::ca-certificates-2019.3.9~ --> anaconda::ca-certificates-2019.5.15-0
  openssl            conda-forge::openssl-1.1.1b-h14c3975_1 --> anaconda::openssl-1.1.1-h7b6447c_0

The following packages will be SUPERSEDED by a higher-priority channel:

  certifi                                       conda-forge --> anaconda


Proceed ([y]/n)?

Downloading and Extracting Packages
mock-3.0.5           | 47 KB     | ################################################################################ | 100%
ca-certificates-2019 | 133 KB    | ################################################################################ | 100%
_tflow_select-2.1.0  | 2 KB      | ################################################################################ | 100%
cupti-9.0.176        | 1.6 MB    | ################################################################################ | 100%
markdown-3.1.1       | 113 KB    | ################################################################################ | 100%
tensorflow-gpu-1.13. | 2 KB      | ################################################################################ | 100%
c-ares-1.15.0        | 98 KB     | ################################################################################ | 100%
tensorflow-base-1.13 | 271.6 MB  | ################################################################################ | 100%
tensorboard-1.13.1   | 3.3 MB    | ################################################################################ | 100%
gast-0.2.2           | 138 KB    | ################################################################################ | 100%
grpcio-1.16.1        | 1.1 MB    | ################################################################################ | 100%
tensorflow-estimator | 205 KB    | ################################################################################ | 100%
astor-0.7.1          | 43 KB     | ################################################################################ | 100%
tensorflow-1.13.1    | 4 KB      | ################################################################################ | 100%
certifi-2019.3.9     | 155 KB    | ################################################################################ | 100%
absl-py-0.7.1        | 157 KB    | ################################################################################ | 100%
termcolor-1.1.0      | 7 KB      | ################################################################################ | 100%
werkzeug-0.15.4      | 262 KB    | ################################################################################ | 100%
openssl-1.1.1        | 5.0 MB    | ################################################################################ | 100%
keras-preprocessing- | 36 KB     | ################################################################################ | 100%
keras-applications-1 | 33 KB     | ################################################################################ | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
crestle@cju6sy2ow001b0130m6d4e1th-87965598c-g4cnr:~$

```

### GPU status 
**nvidia-smi**

```

crestle@cju6sy2ow001b0130m6d4e1th-87965598c-g4cnr:~$ nvidia-smi
Tue Jun 11 17:11:01 2019
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.111                Driver Version: 384.111                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P4            Off  | 00000000:00:07.0 Off |                    0 |
| N/A   31C    P8     6W /  75W |      0MiB /  7606MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
crestle@cju6sy2ow001b0130m6d4e1th-87965598c-g4cnr:~$

```

### Different types of filters in CNNs


* Right Sobel Filter
* Top Sobel Filter
* Blur Filter



Image|13x8
---|---
Filter| 3x3
Filter Map| 11x6

Output Height = (Input Height - Filter Height )/ (Row Stride )+1   
Output Width = (Input Width - Filter Width )/ (Row Stride )+1

Filter has same number chanels are the Image

Image |Filter |Feature Map
---|---|---|---
720 x 960 x 3 | 3 x 3 x 3 | 718 x 958


## Padding in CNN

Valid
Same

Convolutional Layers: To pad or not to pad?
There are couple of reasons padding is important:

It's easier to design networks if we preserve the height and width and don't have to worry too much about tensor dimensions when going from one layer to another because dimensions will just "work".
It allows us to design deeper networks. Without padding, reduction in volume size would reduce too quickly.
Padding actually improves performance by keeping information at the borders.
Quote from Stanford lectures: "In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be “washed away” too quickly." - source


https://stats.stackexchange.com/questions/246512/convolutional-layers-to-pad-or-not-to-pad



## Pooling
Used for demensionality reduction

__Max Pooling__ take max value of a from a pool of pixels.   
__Average Pooling__ take average value of a from a pool of pixels.
Give 3d output for a 3d image

#### Special type of pooling
Give 1d output for a 3d input.
___Global Average Pooling___ Take average of each chanel and get an average of the layer.

___Global Max Pooling___ Take average of each chanel and get an average of the layer.


Questions
1. Which of the following layer is a downsampling layer?
2. Pooling helps in reducing the dimension of the input image.
3. What are the different types of pooling that can be applied to a layer?
4. Consider the below 2-D Input Image of shape (4 X 4):


 We have applied Average Pooling on the above image with stride = 2. What will be the Output Image?
5. Which of the following pooling strategy can be used to convert a 3-D image to 1-D image?


### Architecture of CNN using Filters, Padding and Pooling Layers


What is the correct sequence of a network with CNN architecture?

Input -> CNN -> Fully Connected Neural Network -> Output

Which of the following is/are hyperparameter(s) for a Convolutional Neural Network?

Choose only ONE best answer.

A Number of Convolutional Filters
B Size of Pooling
C Stride
D Number of Pooling Filters
E All of the above



## Transfer Learning
The three major Transfer Learning scenarios look as follows:

* **ConvNet as fixed feature extractor.** Take a ConvNet pretrained on ImageNet, remove the last
fully-connected layer (this layer’s outputs are the 1000 class scores for a different task like
ImageNet), then treat the rest of the ConvNet as a fixed feature extractor for the new dataset. We call these features _CNN codes._ It is important
for performance that these codes are ReLUd (i.e. thresholded at zero) if they were also thresholded
during the training of the ConvNet on ImageNet (as is usually the case). Once you extract the codes for all images, train a linear classifier (e.g. Linear SVM or Softmax classifier) for the new
dataset.   
* **Fine-tuning the ConvNet.** The second strategy is to not only replace and retrain the classifier on top of the ConvNet on the new dataset, but to also fine-tune the weights of the pretrained network by continuing the backpropagation. It is possible to fine-tune all the layers of the ConvNet, or it’s possible to keep some of the earlier layers fixed (due to overfitting concerns) and only fine-tune some higher-level portion of the network. This is motivated by the observation that the earlier features of a ConvNet contain more generic features (e.g. edge detectors or color blob detectors) that should be useful to many tasks, but later layers of the ConvNet becomes progressively more specific to the details of the classes contained in the original dataset. In case of ImageNet for example, which contains many dog breeds, a significant portion of the representational power of the ConvNet may be devoted to features that are specific to differentiating between dog breeds.
* **Pretrained models.** As deep CNN take lot of time to train  For example, the Caffe library has a [Model
Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo) where people share where people share
their network weights.

__When and how to fine-tune?__ How do you decide what type of transfer learning you should perform on a new dataset? This is a function of several factors, but the two most important ones are the size of the new dataset (small or big), and its similarity to the original dataset (e.g. ImageNet-like in terms of the content of images and the classes, or very different, such as microscope images). Keeping in mind that ConvNet features are more generic in early layers and more original-dataset-specific in later layers, here are some common rules of thumb for navigating the 4 major scenarios:

* __New dataset is small and similar to original dataset.__ Since the data is small, it is not a good
idea to fine-tune the ConvNet due to overfitting concerns. Since the data is similar to the original
data, we expect higher-level features in the ConvNet to be relevant to this dataset as well. Hence,
the best idea might be to train a linear classifier on the CNN codes.   
* __New dataset is large and similar to the original dataset__. Since we have more data, we can have
more confidence that we won’t overfit if we were to try to fine-tune through the full network.   
* __New dataset is small but very different from the original dataset.__ Since the data is small, it
is likely best to only train a linear classifier. Since the dataset is very different, it might not
be best to train the classifier form the top of the network, which contains more dataset-specific
features. Instead, it might work better to train the SVM classifier from activations somewhere
earlier in the network.   
* __New dataset is large and very different from the original dataset.__ Since the dataset is very large,
we may expect that we can afford to train a ConvNet from scratch. However, in practice it is very
often still beneficial to initialize with weights from a pretrained model. In this case, we would
have enough data and confidence to fine-tune through the entire network.   


http://cs231n.github.io/transfer-learning/

https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced


"https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner\'s-Guide-To-Understanding-Convolutional-Neural-Networks/"






# CNN Model Performance Limitations and how to overcome

### Less Training Data
#### Data Augumentation

* Popular Augmentation Techniques

  1. featurewise_centre           - Entire Dataset
  * samplewise_centre            - Sample of the Dataset
  * featurewise_std_normalization  - Entire Dataset
  * samplewise_std_normalization   - Sample of the Dataset
  * Rotation
    * rotation_range
  * Shift fractions or pixels
    * width_shift_range
    * height_shift range
  * sheer_range
    done in anti clock wise direction as per the number of degrees specified
  * zoom_range
  * Flipping
    * horizontal_flip
    * Vertical_flip
    * Transpose
  * Add Noise to the data
    * **Gaussian Noise**
Over-fitting usually happens when your neural network tries to learn high frequency features (patterns that occur a lot) that may not be useful. Gaussian noise, which has zero mean, essentially has data points in all frequencies, effectively distorting the high frequency features. This also means that lower frequency components (usually, your intended data) are also distorted, but your neural network can learn to look past that. Adding just the right amount of noise can enhance the learning capability.

    * Salt and Pepper noise
    Salt and Pepper noise refers to addition of white and black dots in the image.
  * Lighting condition
  * Perspective transform:   
      In perspective transform, we try to project image from a different point of view.

  ### Advanced Augmentation Techniques
  Conditional GANs to the rescue!
  Neural style transfer


### High Variation in Data

1. Images have different dimensions
* resize
load_images :keras
resize : skimage

2. Images have same dimensions but different pixel value ranges
Rescale : Divide all pixes with the maximum Pixel Value



### Overfitting
Very high performance in training data but low performance in validation set.

* Dropout -  
* Early Stopping

### Undefitting
* Hyperparameter Tuning
  * Hidden Layers    by  Increasing the number of Hidden Layers
  * Neurons in a Layer  by  Increasing the number of Neurons in the layers
  * Epochs  by  Increasing the number of Epochs  
  * Optimizer by trying different Optimizers
  * Data Related Issues
    by Scaling the data from -1 to +1

### Training Time too high
A neural network takes too much training time to train. The Training and the validation error are in
 sync.
* Change in Data Distribution across layers
Hidden activation changes are during back propagation called _internal covariant shift._
_Batch Normalization_ is added after linear transformation to prevent shifting of the weights of the
hidden layers.
Batch normalization sets the mean to zero and variance as One.

What is Internal Covariate shift in Deep Learning?
Change in Data Distribution across deeper layers of the network
One of the reasons due to which Neural Networks especially CNNs tend to take a lot of time to
converge

### No appropriate Architecture for the problem

* New problem
* Not enough data

Most computer vision problems can be broken down into two categories <mark>Classification/Regression</mark> and
<mark>Detection / Segmentation</mark>.
Some of the popular publically available solutions to these problems are :-


Classification/Regression | Detection / Segmentation
---|---
VGG16 | Mask R-CNNs
VGG19 | YOLO v3
ResNet | SSD
Xception | RetinaNet
Inception | Deep Lab
NASNet|

#### VGG16
** VGG16 ** First NN to reach accuracy close to human vision (5%)  i.e. accuracy of 7.4 %.
It is a Layered NN  Architecture.
First 13 Layers are CNN with pooling layers in the middle after 2, 4, 7 , 10 and 13th Layers.
The last three being the Dense Layers.

#### GoogLeNet (Inception V1)

#### ResNet50  
Contains Identity block and convolution block?

### Evaluation Metrics for the problem

* **Accuracy** - Ratio of Correct vs Total Number of Predictions
* **Recall** - When we want to focus on false Negatives
* **Precision** - When we want to focus false Positive
* **F1 Score** - When we want to focus on False Positive and False Negative. It is HM of Precision and
Recall
* **AUC-ROC Curve** - AUC-ROC represents the Area Under the Curve between the ratio of TPR(True Positive
Rate) and FPR(False Positive Rate)Works on Probabilities Works on Probabilities
* **Log Loss** - Penalize a model in exponential form on the basis of the confidence of the model.

### Evaluation Metrics for Regression Task
* **MAE** <mark>mean_absolute_error</mark>
* **RMSE** <mark>mean_squared_error</mark>
* **Cosine Similarity** <mark>cosine_proximity</mark>



## List of discrete distributions
Maximum Likely Estimator
  Likelihood function  
  Poisson distribution
  Bernoulli distribution
  Binomial distribution
  Exponential distribution
  Gamma distribution
  Probability density function

## Types of Machine Learning Problems
1. Image Classification
  * Binary Classification
  * Multi class Classification
  * **Approach :-**
    * Read the image as numpy Array
    * Build NN Model
    * Apply Sigmoid/Softmax Activation Function at the last layer
* Image Regression
  * Age Prediction
  * Counting Occurances
    * Number of Participants in a Hall
    * **Approach**
      * Read the image as numpy Array
      * Build NN Model
      * Apply Linear Activation Function at the last layer
* Object Detection
 * Types
    * Single Object  Detection
    * Multiple Object Detection - Same class
    * Multiple Object Detection - Multiple Classes
  * **Approach**
    * Classification and Regression Problem after dividing image in patches.
* Action Recognition
  * **Approach**
    * Object Detection
    * Action Recognition
* Image Captioning
  * **Approach :-**
    * Action Recognition and Text Summarization
* Image Segmentation
  * **Approach**
    * Pixel Level Classification on patch of pixels
#### Approach to solve Computer Vision problems

Computer Vision Problem | Classification | Regrression
---|---|---
Image Classification |Yes |No
Image Regression |No |Yes
Object Detection |Yes |Yes
Action Recognition |Yes |No
Image Captioning |Yes |No
Image Segmentation |Yes |Yes
