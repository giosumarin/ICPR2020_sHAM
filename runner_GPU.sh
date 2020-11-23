rm -f nets/GPU.py
cd nets/
ln -s GPU_on.py GPU.py

cd ..
./runner.sh
