import chainer
from PIL import Image
import csv
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

#↓の変数が0だとオール、1だとハーフ。
labelingFlag = 1
#ラベルセット
def label_set(left,right):
    if(labelingFlag == 0):
        if(((misalignment * i) >= left) and (cutSize + (misalignment * i)) <= right):
            return True
        else:
            return False
    else:
        if(((misalignment * i) >= left and (cutSize +(misalignment * i) <= right))or((misalignment * i) >= (left-cutSize/2) and (cutSize +(misalignment * i) <= (right-cutSize/2)))or((misalignment * i) >= (left+cutSize/2) and (cutSize +(misalignment * i) <= (right+cutSize/2)))):
            return True
        else:
            return False

test = LabeledImageDataset('./test/cut_set.txt',root='./')
#切り取りサイズ(px)
cutSize = 10
#一度にズラす値(px)
misalignment = 5

imageWidth = (350-cutSize)/misalignment
imageHeight = (400-cutSize)/misalignment

with open('blood_test.csv') as f:
    reader = csv.reader(f)
    csvResult = [row for row in reader]

teahcerAverage = []
for j in range(len(csvResult)):
  tmp = []
  for i in range(int(imageWidth)):
    teacherRight = int(csvResult[j][1])
    teacherLeft = int(csvResult[j][0])
    if(label_set(teacherLeft,teacherRight)):
      tmp.append(1)
    else:
      tmp.append(0)
  teahcerAverage.append(tmp)

print(teahcerAverage)