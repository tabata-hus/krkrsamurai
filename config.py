afterImageElement = open('after/cut_set.txt','r').readlines()
testImageElement = open('test/cut_set.txt','r').readlines()
#モデル作成時に学習データとして使う数(%)
modelCreateLearningPercentage = 0.95
batchSize = 128
middleLayer = 1000
#gpu_id=0/-1(GPU/CPU)
gpu_id = -1
#学習回数
maxEpoch = 100
#ラベル付けするとき、画像内の何％が血管なら血管と判定するかの基準 1/0 (50%/100%)
labelingFlag = 1
#切り取りサイズ(px)
cutSize = 10
#一度にズラす値(px)
misalignment = 5
#縦列平均取るときの閾値
threshold = 0.5