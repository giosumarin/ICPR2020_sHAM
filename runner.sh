python -m venv venv_compr
source venv_compr/bin/activate
pip install --upgrade pip
pip install -r env_packages.txt

cd compressionNN_package/
pip install -e ./
cd ..

pip install matplotlib
pip install seaborn
pip install pandas


cd nets/DeepDTA/KIBA
./run_deepdtakiba.sh
cd ../../../

cd nets/DeepDTA/DAVIS
./run_deepdtadavis.sh
cd ../../../

cd nets/VGG19-CIFAR10
./run_vgg19cifar.sh
cd ../../

cd nets/VGG19-MNIST
./run_vgg19mnist.sh
cd ../../

cd time_space/


python testing_time_space_deepdta.py -t weightsharing -d ../nets/DeepDTA/KIBA/weightsharing/ -m ../nets/DeepDTA/KIBA/deepDTA_kiba.h5 -s kiba -q 0
python testing_time_space_deepdta.py -t pruningweightsharing -d ../nets/DeepDTA/KIBA/pruningweightsharing/ -m ../nets/DeepDTA/KIBA/deepDTA_kiba.h5 -s kiba -q 0
python testing_time_space_deepdta.py -t weightsharing -d ../nets/DeepDTA/KIBA/stochastic/ -m ../nets/DeepDTA/KIBA/deepDTA_kiba.h5 -s kiba -q 1
python testing_time_space_deepdta.py -t pruningweightsharing -d ../nets/DeepDTA/KIBA/pruning_stochastic/ -m ../nets/DeepDTA/KIBA/deepDTA_kiba.h5 -s kiba -q 1


python testing_time_space_deepdta.py -t weightsharing -d ../nets/DeepDTA/DAVIS/weightsharing/ -m ../nets/DeepDTA/DAVIS/deepDTA_davis.h5 -s davis -q 0
python testing_time_space_deepdta.py -t pruningweightsharing -d ../nets/DeepDTA/DAVIS/pruningweightsharing/ -m ../nets/DeepDTA/DAVIS/deepDTA_davis.h5 -s davis -q 0
python testing_time_space_deepdta.py -t weightsharing -d ../nets/DeepDTA/DAVIS/stochastic/ -m ../nets/DeepDTA/DAVIS/deepDTA_davis.h5 -s davis -q 1
python testing_time_space_deepdta.py -t pruningweightsharing -d ../nets/DeepDTA/DAVIS/pruning_stochastic/ -m ../nets/DeepDTA/DAVIS/deepDTA_davis.h5 -s davis -q 1




python testing_time_space.py -t weightsharing -d ../nets/VGG19-CIFAR10/weightsharing/ -m ../nets/VGG19-CIFAR10/retrain.h5 -s cifar10_vgg -q 0
python testing_time_space.py -t pruningweightsharing -d ../nets/VGG19-CIFAR10/pruningweightsharing/ -m ../nets/VGG19-CIFAR10/retrain.h5 -s cifar10_vgg -q 0
python testing_time_space.py -t weightsharing -d ../nets/VGG19-CIFAR10/stochastic/ -m ../nets/VGG19-CIFAR10/retrain.h5 -s cifar10_vgg -q 1
python testing_time_space.py -t pruningweightsharing -d ../nets/VGG19-CIFAR10/pruning_stochastic/ -m ../nets/VGG19-CIFAR10/retrain.h5 -s cifar10_vgg -q 1


python testing_time_space.py -t weightsharing -d ../nets/VGG19-MNIST/weightsharing/ -m ../nets/VGG19-MNIST/VGG19MNIST.h5 -s mnist -q 0
python testing_time_space.py -t pruningweightsharing -d ../nets/VGG19-MNIST/pruningweightsharing/ -m ../nets/VGG19-MNIST/VGG19MNIST.h5 -s mnist -q 0
python testing_time_space.py -t weightsharing -d ../nets/VGG19-MNIST/stochastic/ -m ../nets/VGG19-MNIST/VGG19MNIST.h5 -s mnist -q 1
python testing_time_space.py -t pruningweightsharing -d ../nets/VGG19-MNIST/pruning_stochastic/ -m ../nets/VGG19-MNIST/VGG19MNIST.h5 -s mnist -q 1
