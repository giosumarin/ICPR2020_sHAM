# Compression strategies and space-conscious representations for deep neural networks
This repository contains the code allowing to reproduce the results described in G. Marinò et al.,
_Compression strategies and space-conscious representations for deep neural networks_, accepted at
the forthcoming [ICPR](https://www.micc.unifi.it/icpr2020/) conference. The original contribution
is [available](ICPR2020_sHAM.pdf) for reviewing purposes of a companion paper currently submitted at
the [RRPR](https://rrpr2020.sciencesconf.org/) conference.


## Getting Started

### Prerequisites

* Install `python3`, `python3-pip` and `python3-venv` (Debian 10.6)
* Make sure that `python --version` starts by 3 or execute `alias python='pyhton3'` in the shell before executing `runner.sh`.
* For CUDA configuration follow https://www.tensorflow.org/install/gpu.
* Jupyter Notebook is needed for chart generation.

<!--
tensorflow-gpu==2.2.0 or tensorflow==2.2.0
numpy==1.18.1
scikit-learn==0.22.1
scipy==1.4.1
numba==0.49.1
joblib==0.14.1
matplotlib==3.1.3
Anaconda installation

Installation
Go to compressionNN_package and install the package with setup.py
-->
### Configuration
The library import `import keras.backend.tensorflow_backend as tfback` raises an exception if no GPU is available.
This can be fixed by commenting out all lines in [`nets/GPU.py in nets/`](nets/GPU.py)

The trained VGG models are rather big, so they are not versioned. Rather, they are available for [download](https://mega.nz/folder/yKgU2CYD#-Kf3FGZinDe5T6HgLOjxnw).
Once downloaded, `VGG19-CIFAR/retrain.h5 ` should be moved in [`nets/VGG19-CIFAR`](nets/VGG19-CIFAR) and `VGG19-MNIST/VGG19MNIST.h5` should be moved in [`nets/VGG19-MNIST`](nets/VGG19-MNIST).


## Usage
1. Give execute permissions and run [`runner.sh`](runner.sh).
2. Open the [`plot_from_file.ipynb`](plot\_from\_file.ipynb) notebook, change `directory_res` as described and evaluate all cells to produce the charts. Currently, the notebook reads the results from `results_just_runned/` and saves the produced graphs in `plots_results/`. The same notebook also produces the results illustrated in the tables of the paper, organized in terms of compression method and neural networks.
