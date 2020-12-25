import chainer
#import cupy
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


train_val = LabeledImageDataset('./after/cut_set.txt',root='./')
test = LabeledImageDataset('./test/cut_set.txt',root='./')



def transform(data):
  img,label = data
  img = img/255
  return img,label

train_val = chainer.datasets.TransformDataset(train_val,transform)
test = chainer.datasets.TransformDataset(test,transform)

train, valid = split_dataset_random(train_val, int(len(cf.afterImageElement)*cf.modelCreateLearningPercentage), seed=0)

print('Training dataset size:', len(train))
print('Validation dataset size:', len(valid))


#バッチサイズが増えると学習回数が増える（過学習の危険あり）
#レイヤー数も増やしたほうが精度は上がるだろう（過学習を恐れろ）
batchsize = cf.batchSize

train_iter = iterators.SerialIterator(train, batchsize)
valid_iter = iterators.SerialIterator(valid, batchsize, repeat=False, shuffle=False)
test_iter = iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)



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

net = MLP()

if gpu_id >= 0:
    net.to_gpu(gpu_id)

#print('1つ目の全結合相のバイアスパラメータの形は、', net.l1.b.shape)
#print('初期化直後のその値は、', net.l1.b.array)
#print(net.l1.W.array)

optimizer = optimizers.SGD(lr=0.01).setup(net)



max_epoch = cf.maxEpoch

while train_iter.epoch < max_epoch:
    # ---------- 学習の1イテレーション ----------
    train_batch = train_iter.next()
    x, t = concat_examples(train_batch, gpu_id)
    # 予測値の計算
    y = net(x)

    # ロスの計算
    loss = F.softmax_cross_entropy(y, t)

    # 勾配の計算
    net.cleargrads()
    loss.backward()

    # パラメータの更新
    optimizer.update()
    # --------------- ここまで ----------------

    # 1エポック終了ごとにValidationデータに対する予測精度を測って、
    # モデルの汎化性能が向上していることをチェックしよう
    if train_iter.is_new_epoch:  # 1 epochが終わったら

        # ロスの表示
        print('epoch:{:02d} train_loss:{:.04f} '.format(
            train_iter.epoch, float(to_cpu(loss.data))), end='')

        valid_losses = []
        valid_accuracies = []
        while True:
            valid_batch = valid_iter.next()
            x_valid, t_valid = concat_examples(valid_batch, gpu_id)

            # Validationデータをforward
            with chainer.using_config('train', False), \
                    chainer.using_config('enable_backprop', False):
                y_valid = net(x_valid)

            # ロスを計算
            loss_valid = F.softmax_cross_entropy(y_valid, t_valid)
            valid_losses.append(to_cpu(loss_valid.array))

            # 精度を計算
            accuracy = F.accuracy(y_valid, t_valid)
            accuracy.to_cpu()
            valid_accuracies.append(accuracy.array)
            if valid_iter.is_new_epoch:
                valid_iter.reset()
                break

        print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
            np.mean(valid_losses), np.mean(valid_accuracies)))
# テストデータでの評価
test_accuracies = []
while True:
    test_batch = test_iter.next()
    x_test, t_test = concat_examples(test_batch, gpu_id)

    # テストデータをforward
    with chainer.using_config('train', False), \
            chainer.using_config('enable_backprop', False):
        y_test = net(x_test)

    # 精度を計算
    accuracy = F.accuracy(y_valid, t_valid)
    accuracy.to_cpu()
    test_accuracies.append(accuracy.array)

    if test_iter.is_new_epoch:
        test_iter.reset()
        break

print('test_accuracy:{:.04f}'.format(np.mean(test_accuracies)))

serializers.save_npz('my_mnist.model', net)