import chainer
from PIL import Image
import config as cf
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
labelingFlag = cf.labelingFlag
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
cutSize = cf.cutSize
#一度にズラす値(px)
misalignment = cf.misalignment

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

def transform(data):
  img,lable = data
  img = img/255
  return img,lable


test = chainer.datasets.TransformDataset(test,transform)


def reset_seed(seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)
reset_seed(0)

class MLP(chainer.Chain):

    def __init__(self, n_mid_units=cf.middleLayer, n_out=2):
        super(MLP, self).__init__()
        # パラメータを持つ層の登録
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(n_mid_units, n_mid_units)
            self.l3 = L.Linear(n_mid_units, n_out)

    def __call__(self, x):
        # データを受け取った際のforward計算を書く
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

gpu_id = cf.gpu_id  # CPUを用いる場合は、この値を-1にしてください


loadModel = 'my_mnist.model'
infer_net = MLP()
serializers.load_npz(loadModel,infer_net)
imageResult = np.array([])
for i in range(len(cf.testImageElement)):
  # 分類したいデータをモデルに渡します
  predict_data,predict_label = test[i]
  predict_data = predict_data[None, ...]
  #
  predict = infer_net(predict_data)
  result= predict.array
  #print(result)
  probable_label = result.argmax(axis=1)
  imageResult = np.append(imageResult,probable_label[0])
  #print('Most likely :', probable_label[0])
  #print('true or false')
  # if(probable_label[0] != test[i][1]):
  #       #print('false')
  #       returnNum = i//68
  #       yPoint = 5 * returnNum
  #       if(yPoint > 400):
  #             yPoint -= 400
  #       print('x:',5*(i%68),'y:',yPoint)

imageResult = imageResult.reshape([-1,int(imageWidth)])
tmpAverage = []
resultAverage = []
print(np.shape(imageResult))
tempNum = 0
#画像枚数分繰り返すfor jj
for jj in range(np.shape(imageResult)[0]//int(imageHeight)):
      #
      for j in range(np.shape(imageResult)[1]):
            tmp = 0
            for i in range(int(imageHeight)*jj,int(imageHeight)*(jj+1)):
                  tmp += imageResult[i,j]
            tmp /= int(imageHeight)
            tmpAverage.append(tmp)
      #print(tmpAverage)
      resultAverage.append(tmpAverage)
      tmpAverage = []

print(teahcerAverage)
print(resultAverage)

threshold = cf.threshold
print("threshold is",threshold)
for j in range(len(csvResult)):
  innerDiameter = []
  for i in range(int(imageWidth)):
    if(resultAverage[j][i] >= threshold):
      innerDiameter.append(i)
  print("InnerDiameter",misalignment*innerDiameter[0],cutSize + misalignment * innerDiameter[len(innerDiameter)-1])
  print("TeacherInnerDiameter",csvResult[j][0],csvResult[j][1])

# for j in range(np.shape(imageResult)[1]):
#       tmp = 0
#       for i in range(78):
#             tmp += imageResult[tempNum,j]
#             tempNum += 1
#       tmp /= np.shape(imageResult)[1]
#       lastResult.append(tmp)


