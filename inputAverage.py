import chainer
#import cupy
from matplotlib import pyplot as plt
import config as cf
import random
import numpy
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer.cuda import to_cpu
from chainer.datasets import mnist
from chainer.datasets import LabeledImageDataset
from chainer.datasets import split_dataset_random
from chainer.datasets import TransformDataset
from chainer.dataset import concat_examples
from chainer import iterators
from chainer import serializers

def transform(data):
  img,label = data
  img = img/255
  return img,label

train_val = LabeledImageDataset('./after/cut_set.txt',root='./')
train_val = chainer.datasets.TransformDataset(train_val,transform)

#正規化したあとの話なんだけど、白が１で黒が０だった


inputAverageList = []
inputAverageListThreshold = []
threshold = 0.12
num = 10000
inputAverageX = []
inputAverageY = []
# for i in range(10):
#     print(train_val[i][0][0])
#     for j in range(10):
#         for x in range(10):

for i in range(num):
    tmp = 0
    for j in range(10):
        for item in train_val[i][0][0][j]:
            tmp += item
    inputAverageList.append([tmp/100,train_val[i][1]])
    inputAverageListThreshold.append([tmp/100,1 if tmp/100 <= threshold else 0])
    inputAverageX.append(tmp/100)
    inputAverageY.append(train_val[i][1])
    #print(tmp/100,train_val[i][1])

#print(inputAverageList)


# plt.scatter(inputAverageX,inputAverageY)
# plt.show()


count = 0
for i in range(num):
  if(inputAverageList[i][1] == inputAverageListThreshold[i][1]):
    count += 1
print("Correct answer rate",count/num)
print("Threshold",threshold)