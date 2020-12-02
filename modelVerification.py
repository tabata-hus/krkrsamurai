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


test = LabeledImageDataset('./test/cut_set.txt',root='./')
#切り取りサイズ(px)
cutSize = 10
#一度にズラす値(px)
misalignment = 5

imageWidth = (350-cutSize)/misalignment
imageHeight = (400-cutSize)/misalignment

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

    def __init__(self, n_mid_units=1000, n_out=2):
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

gpu_id = -1  # CPUを用いる場合は、この値を-1にしてください


loadModel = 'my_mnist.model'
infer_net = MLP()
serializers.load_npz(loadModel,infer_net)
imageResult = np.array([])
for i in range(21216):
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

imageResult = imageResult.reshape([-1,68])
lastResult = []
print(np.shape(imageResult))
tempNum = 0
for jj in range(np.shape(imageResult)[0]//78):
      for j in range(np.shape(imageResult)[1]):
            tmp = 0
            for i in range(78*jj,78*(jj+1)):
                  tmp += imageResult[i,j]
            tmp /= 78
            lastResult.append(tmp)
      print(lastResult)
      lastResult = []




# for j in range(np.shape(imageResult)[1]):
#       tmp = 0
#       for i in range(78):
#             tmp += imageResult[tempNum,j]
#             tempNum += 1
#       tmp /= np.shape(imageResult)[1]
#       lastResult.append(tmp)


