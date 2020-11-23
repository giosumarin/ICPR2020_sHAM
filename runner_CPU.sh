rm -f nets/GPU.py
cd nets/
ln -s GPU_off.py GPU.py

cd ..
./runner.sh
