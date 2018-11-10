##Caffe

## Installation
Instructions for Ubuntu 16.04.5 LTS  
Hardware: NVIDIA JETSON TX2  
Reference: http://caffe.berkeleyvision.org/install_apt.html

```
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install python
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev
```
## Compilation
Reference: http://caffe.berkeleyvision.org/installation.html#compilation
```
mkdir .local/install
cd .local/install
git clone https://github.com/BVLC/caffe.git
cd caffe
cd python/
for req in $(cat requirements.txt); do sudo pip install $req; done
cd ..
cp Makefile.config.example Makefile.config
make all
```
If the last command fails with: **fatal error: hdf5.h: No such file or directory**
```
vi Makefile.config
```
Add ```/usr/include/hdf5/serial/```  at the end of line **INCLUDE_DIRS**
Exit vi
```
find /usr/lib -name hdf5
```
Copy the return string, in my case: **/usr/lib/aarch64-linux-gnu/hdf5**
```
vi Makefile.config
```
Add the return string + **/serial/** to **LIBRARY_DIRS** in my case:
```
/usr/lib/aarch64-linux-gnu/hdf5/serial/ 
```
Exit vi
```
make clean
make all
make test
make runtest
```






