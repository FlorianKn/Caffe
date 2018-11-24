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
Steps:
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
mkdir dataset
cd dataset  
```
Copy kfkd.py from **dataset/** located in this repository. The script converts CSV to hd5.     
```
python kfkd.py
```
*Note: the python script was adapted from http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/*

### 2.Model definition  
The architecture of a network is defined in a **.prototxt**. An example of an architecture can be seen below:  
```
name: "Arch_Baseline"
layer {
  name: "Input"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "dataset/train_data_list.txt"
    batch_size: 128
    shuffle: true
  }
}
layer {
  name: "ip"
  type: "InnerProduct"
  bottom: "data"
  top: "ip"
  inner_product_param {
    num_output: 30
  }
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "ip"
  bottom: "label"
  top: "loss"
}
```
*Find this file in models/Arch_Baseline/baseline.prototxt*
#### 2.1 Visualizing architecture [Optional]  
If you want to print your network architecture you have to configure pycaffe. Therefore the **Makefile.config** needs to be adapted.  
*Note: I use python2.7.12 and numpy needs to be installed.*  
Find the paths to **Python.h** and **numpy/arrayobject.h** on your machine. Add them to **PYTHON_INCLUDE** in your Makefile.config. See my result below:  
![Alt text](/Screenshots/pycaffe_01.png?raw=true "pyC_01")  
Find the path to libpythonX.X.so and add it to **PYTHON_LIB**. See my result below:  
![Alt text](/Screenshots/pycaffe_02.png?raw=true "pyC_02")  
Save those changes and run the following commands:
```
make clean
make all
make test
make runtest
```  
Set the PYTHONPATH variable in your ~/.bashrc to the caffe python with ```export PYTHONPATH=<caffe-home>/python:$PYTHONPATH```. In my case ```export PYTHONPATH=/home/nvidia/.local/install/caffe/python:$PYTHONPATH```  

Install missing dependencies and compile pycaffe, therefore run:
```
for req in $(cat python/requirements.txt); do sudo pip install $req; done
make pycaffe
```
*Reference to configure pycaffe: http://installing-caffe-the-right-way.wikidot.com/start*

To print the architecture pydot and graphviz needs to be installed.
```
sudo pip install pydot
sudo apt-get install graphviz
```
Command for printing is ```python /my/directory/caffe/python/draw_net.py /my/directory/caffe/my/model/myModel.prototxt /my/directory/caffe/myImages/myModel.png```.  
In my case ```python /home/nvidia/.local/install/caffe/python/draw_net.py /home/nvidia/.local/install/caffe/models/Arch_Baseline/baseline.prototxt /home/nvidia/.local/install/caffe/architectureImg/baseline.png```.  
The image below shows the printed architecture of the .prototxt defined above:
![Alt text](/architectureImg/baseline.png?raw=true "bl_02") 
### 3.Solver definition
The solver orchestrates model optimization by coordinating the network’s forward inference and backward gradients to form parameter updates that attempt to improve the loss.  
Reference: http://caffe.berkeleyvision.org/tutorial/solver.html

See my solver below:
```
net: "models/Arch_Baseline/baseline.prototxt"

#Nesterov’s Accelerated Gradient
type: "Nesterov"
test_iter: 2
# testing every 100 iterations
test_interval: 100

# learning rate
base_lr: 0.001
lr_policy: "step"
gamma: 0.2
#drop learning rate every 1000 iterations
stepsize: 1000
momentum: 0.9

weight_decay: 0.01

display: 100
#train for 3000 iterations
max_iter: 3000

snapshot: 500
snapshot_prefix: "/home/nvidia/.local/install/caffe/models/Arch_Baseline/"
solver_mode: GPU

test_compute_loss: true
```
*Find this file in /models/Arch_Baseline/solver.prototxt*
### 4.Model training
