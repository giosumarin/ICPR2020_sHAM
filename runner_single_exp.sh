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


cd nets/DeepDTA/DAVIS
python pruning_weightsharing.py 0 0.001 80 32 2 2 32 #LOOK TABLE 3 OF PAPER
cd ../../../

cd time_space/

python testing_time_space_deepdta.py -t pruningweightsharing -d ../nets/DeepDTA/DAVIS/pruningweightsharing/ -m ../nets/DeepDTA/DAVIS/deepDTA_davis.h5 -s davis -q 0


python testing_time_space_deepdta_saving.py -t pruningweightsharing -d ../nets/DeepDTA/DAVIS/pruningweightsharing/ -m ../nets/DeepDTA/DAVIS/deepDTA_davis.h5 -s davis -q 0

