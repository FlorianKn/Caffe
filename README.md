# Caffe - Facial Keypoint Detection

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
## Building a Network  
Steps:
1. Data preperation
2. Model defintion
3. Solver defintion
4. Model training
5. Model testing

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
Copy fkp.py from **dataset/** located in this repository. The script converts CSV to hd5. Add the kaggle dataset to your folder, too.     
```
python fkp.py
```
*Note: the python script was adapted from https://github.com/olddocks/caffe-facialkp*

### 2.Model definition  
The architecture of a network is defined in a **.prototxt**. An example of an architecture can be seen below:  
```
name: "example_Network"
layer {
  name: "MyData"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "dataset/train.txt"
    batch_size: 64
    shuffle: true
  }
  include: { phase: TRAIN }
}
layer {
  name: "MyData"
  type: "HDF5Data"
  top: "data"
  top: "label"
  hdf5_data_param {
    source: "dataset/test.txt"
    batch_size: 100
  }
  include: { phase: TEST }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  #blobs_lr: 1
  #blobs_lr: 2
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "dropout1"
  type: "Dropout"
  bottom: "pool1"
  top: "pool1"
  dropout_param {
    dropout_ratio: 0.1
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  #blobs_lr: 1
  #blobs_lr: 2
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "dropout2"
  type: "Dropout"
  bottom: "pool2"
  top: "pool2"
  dropout_param {
    dropout_ratio: 0.3
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  #blobs_lr: 1
  #blobs_lr: 2
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE

    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
 }
 layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "dropout3"
  type: "Dropout"
  bottom: "conv3"
  top: "conv3"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc5"
  type: "InnerProduct"
  bottom: "conv3"
  top: "fc5"
  #blobs_lr: 1
  #blobs_lr: 2
  #weight_decay: 1
  #weight_decay: 0
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "drop4"
  type: "Dropout"
  bottom: "fc5"
  top: "fc5"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "fc5"
  top: "fc6"
  #blobs_lr: 1
  #blobs_lr: 2
  #weight_decay: 1
  #weight_decay: 0
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 30
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "fc6"
  bottom: "label"
  top: "loss"
}
```
*Note: find this file in models/Example/example.prototxt. Reference: https://github.com/qiexing/caffe-regression/blob/master/kaggle_prototxt/fkp_net.prototxt*
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
In my case ```python /home/nvidia/.local/install/caffe/python/draw_net.py /home/nvidia/.local/install/caffe/models/Example/example.prototxt /home/nvidia/.local/install/caffe/architectureImg/example.png```.  
The image below shows the printed architecture of the .prototxt defined above:
![Alt text](/architectureImg/example.png?raw=true "ex_01") 
### 3.Solver definition
The solver orchestrates model optimization by coordinating the network’s forward inference and backward gradients to form parameter updates that attempt to improve the loss.  
Reference: http://caffe.berkeleyvision.org/tutorial/solver.html

See my solver below:
```
# The training protocol buffer definition
net: "models/Example/example.prototxt"
test_iter: 1
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.001
weight_decay : 0.0005
solver_type : NESTEROV
momentum: 0.9
# The learning rate policy
lr_policy: "fixed"
gamma: 0.0001
power: 0.75
stepsize: 300
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 3000
# snapshot intermediate results
snapshot: 1000
snapshot_prefix: "/home/nvidia/.local/install/caffe/models/Example/"
# solver mode: CPU or GPU
solver_mode: GPU
```
*Find this file in /models/Example/solver.prototxt*.   
All parameters to configure your network: https://github.com/BVLC/caffe/wiki/Solver-Prototxt
### 4.Model training
Start the training with the following command:
```
/home/nvidia/.local/install/caffe/build/tools/caffe train --solver /home/nvidia/.local/install/caffe/models/Example/solver.prototxt 2>&1 | tee /home/nvidia/.local/install/caffe/models/Example/Ex_train.log
```
*Note: you need to adapt the paths*.  

The ```2>&1 | tee .../Arch_train.log``` part of the command saves screen output
to a log file.
 
### 5.Model testing  
After training the model, predictions on unseen data can be made.  
But in its current state, the network is not designed for deployment. A **deploy.prototxt** has to be implemented. Therefore **example.prototxt** needs to be modified (reference: https://github.com/BVLC/caffe/wiki/Using-a-Trained-Network:-Deploy).  
See my **deploy.prototxt** below:
```
name: "example_Network"
layer {
  name: "data"
  type: "MemoryData"
  top: "data"
  top: "label"
  memory_data_param {
    batch_size: 64 #batch size, so how many prediction you want to do at once. Best is "1", but higher number get better performance
    channels: 1
    height: 96
    width: 96 

  }
}
layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  #blobs_lr: 1
  #blobs_lr: 2
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "dropout1"
  type: "Dropout"
  bottom: "pool1"
  top: "pool1"
  dropout_param {
    dropout_ratio: 0.1
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  #blobs_lr: 1
  #blobs_lr: 2
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 48
    kernel_size: 5
    stride: 1
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 2
    stride: 2
  }
}
layer {
  name: "dropout2"
  type: "Dropout"
  bottom: "pool2"
  top: "pool2"
  dropout_param {
    dropout_ratio: 0.3
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  #blobs_lr: 1
  #blobs_lr: 2
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_size: 3
    stride: 1
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE

    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
 }
 layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "dropout3"
  type: "Dropout"
  bottom: "conv3"
  top: "conv3"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc5"
  type: "InnerProduct"
  bottom: "conv3"
  top: "fc5"
  #blobs_lr: 1
  #blobs_lr: 2
  #weight_decay: 1
  #weight_decay: 0
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 500
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layer {
  name: "drop4"
  type: "Dropout"
  bottom: "fc5"
  top: "fc5"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "fc5"
  top: "fc6"
  #blobs_lr: 1
  #blobs_lr: 2
  #weight_decay: 1
  #weight_decay: 0
  param {
    lr_mult: 10
    decay_mult: 1
  }
  param {
    lr_mult: 20
    decay_mult: 0
  }
  inner_product_param {
    num_output: 30
    weight_filler {
      type: "xavier"
      variance_norm: AVERAGE
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
```
*Note: find this file in models/Example/deploy.prototxt. Reference: https://github.com/qiexing/caffe-regression/blob/master/kaggle_prototxt/fkp_deploy.prototxt*  
Now the network can be tested. To display the facial-keypoints on the images, install this first: ```sudo apt-get install python-gi-cairo```.  
Afterwards run: ```python output.py```  
*Note: output.py was adapted from https://github.com/olddocks/caffe-facialkp*.  
The results are stored in a CSV.  
See an example image below:

![Alt text](/outputImg/Exampleout.png?raw=true "out_01")

## Competition  
I trained and tested three networks: Baseline, LeNet and VGG. Find all files in ```models```.
For training and testing I splitted kaggles training.csv (80% training, 20% testing). The corresponding hd5-files are located in ```dataset```.  
I used the following hyperparameters:  
```
iterations: 250
solver: ADAM
learning rate: 0.001
snapshot: 100
batch size: 32
```  
  
See my results here:  
| Network  | Average loss  | Average duration  |  
|----------|:-------------:|-----------------:|  
| Baseline | 2.482     | 9.112s      |  
| LeNet    | 1.308     | 25.978s     |  
| VGG      | 43.321    | 215.959s    |  
  
More details in ```Competition.pdf```  

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |
