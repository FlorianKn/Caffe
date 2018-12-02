import os
import numpy as np
import math
import sys
import h5py
from matplotlib import pyplot
sys.path.append('/home/nvidia/.local/install/caffe/python')

import caffe
MODEL_FILE = 'models/Example/deploy.prototxt'
PRETRAINED = 'models/Example/solver_iter_15000.caffemodel' 
height = 96




def predictImg(data4D,layername):
    data4DL = np.zeros([data4D.shape[0],1,1,1])   
    net.set_input_arrays(data4D.astype(np.float32),data4DL.astype(np.float32))
    out = net.forward()
    prediction = net.blobs[layername].data
    return prediction
  
def plot_sample(x, y, axis):
    img = x.reshape(height, height)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * height/2 + height/2, y[1::2] * height/2 + height/2, marker='x', s=10)

print os.getcwd()

t = '/home/nvidia/.local/install/caffe/dataset/'
f = h5py.File(t + 'test_data_list.txt','r')
X = f['data'][:]
print X.shape
net=caffe.Net(MODEL_FILE,PRETRAINED,caffe.TEST)
y_pred = predictImg(X,'fc6')

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    plot_sample(X[i], y_pred[i], ax)
pyplot.show()

