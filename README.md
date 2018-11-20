# Caffe

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
cp Makefile.config.example Makefile.config
make all
```
*Note: the for req in ... command took ~60 minutes*  

If the last command fails with: **fatal error: hdf5.h: No such file or directory**
```
vi Makefile.config
```
Add ```/usr/include/hdf5/serial/```  at the end of line **INCLUDE_DIRS**  
The result should look like this:
![Alt text](/Screenshots/compilation_01.png?raw=true "Comp_01")  
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
The result should look like this:
![Alt text](/Screenshots/compilation_02.png?raw=true "Comp_02")
Exit vi
```
make clean
make all
make test
make runtest
```
The result should look like this:  
![Alt text](/Screenshots/compilation_03.png?raw=true "Comp_03")

*Note: make all took ~45 minutes, make runtest took ~60 minutes*  
## Training a Network  
1. Data preperation
2. Model defintion
3. Solver defintion
4. Model training

### 1.Data preperation
Reference (data): https://www.kaggle.com/c/facial-keypoints-detection/data  
To prepare the kaggle dataset (CSV) all data has to be converted to hd5  
Requirements python-dateutil 2.5.0 and sklearn:
```
sudo pip install python-dateutil==2.5.0 
sudo apt install python-sklearn
```
Run kfkd.py to convert the CSV to hd5  
```
cd dataset
python kfkd.py
```
*Note: the python script was adapted from http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/*

### 2.Model definition  
The architecture of a network is defined in a **.prototxt**  
An example of a simple logistic regression classifier can be seen below (Reference: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html):  
```
name: "LogReg"
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  data_param {
    source: "input_leveldb"
    batch_size: 64
  }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "loss"
  type: "SoftmaxWithLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
}
```

#### 2.1 Visualizing architecture [Optional]  
If you want to visualize your architecture you have to configure pycaffe 
