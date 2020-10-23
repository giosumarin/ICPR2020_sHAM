# Compression strategies and space-conscious representations for deep neural networks
Description: available in ICPR2020_sHAM.pdf

## Getting Started
### Prerequisites

Jupyter Notebook for make plots

<!--
tensorflow-gpu==2.2.0 or tensorflow==2.2.0
numpy==1.18.1
scikit-learn==0.22.1
scipy==1.4.1
numba==0.49.1
joblib==0.14.1
matplotlib==3.1.3
Anaconda installation

### Installation
Go to compressionNN_package and install the package with setup.py
-->
### Configuration
If you have problem with "import keras.backend.tensorflow_backend as tfback"
comment all line of GPU.py in nets/
Download trained weights for VGG19 at https://mega.nz/folder/yKgU2CYD#-Kf3FGZinDe5T6HgLOjxnw. Then put VGG19-CIFAR/retrain.h5 in nets/VGG19-CIFAR and VGG19-MNIST/VGG19MNIST.h5 in nets/VGG19-MNIST.


## Usage
Give execute permissions and run runner.sh to perform all the experiments done in the article. After the experiments open the plot\_from\_file.ipynb notebook, change directory_res as indicated to produce the charts.

Currently the notebook reads the results from results\_just\_runned/ and produces the graphs saved in plots_results/
