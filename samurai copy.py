from PIL import Image
import numpy as np
import cv2
import csv

image = []

with open('blood_lern.csv') as f:
    reader = csv.reader(f)
    csvResult = [row for row in reader]

for i in range(len(csvResult)):
        image.append(Image.open('./before/' + csvResult[i][2]))

#すべて含まれてる場合のラベルセット
def label_set(left,right):
    if(((misalignment * i) >= left) and (cutSize + (misalignment * i)) <= right):
        return True
    else:
        return False

path = './after/cut_set.txt'
#切り取りサイズ(px)
cutSize = 10
#一度にズラす値(px)
misalignment = 5

imageWidth = (350-cutSize)/misalignment
imageHeight = (400-cutSize)/misalignment
file = open(path,'w')

num = 0
for jj in range(len(image)):
    for j in range(int(imageHeight)):
        for i in range(int(imageWidth)):
            cut_range = (0+i*misalignment,0+j*misalignment,cutSize+i*misalignment,cutSize+j*misalignment)
            im_crop = image[jj].crop(cut_range)
            im_crop.save('./after/cut_part{}.tif'.format(num))
            if(label_set(int(csvResult[jj][0]),int(csvResult[jj][1]))):
                file.writelines(('./after/cut_part{}.tif'.format(num))+ ' 1\n')
            else:
                file.writelines(('./after/cut_part{}.tif'.format(num))+ ' 0\n')
            num += 1

image = []

with open('blood_test.csv') as f:
    reader = csv.reader(f)
    csvResult = [row for row in reader]

for i in range(len(csvResult)):
        image.append(Image.open('./before/' + csvResult[i][2]))

path = './test/cut_set.txt'
file = open(path,'w')

num = 0
for jj in range(len(image)):
    for j in range(int(imageHeight)):
        for i in range((int(imageWidth))):
            cut_range = (0+i*misalignment,0+j*misalignment,cutSize+i*misalignment,cutSize+j*misalignment)
            im_crop = image[jj].crop(cut_range)
            im_crop.save('./test/cut_part{}.tif'.format(num))
            if(label_set(int(csvResult[jj][0]),int(csvResult[jj][1]))):
                file.writelines(('./test/cut_part{}.tif'.format(num))+ ' 1\n')
            else:
                file.writelines(('./test/cut_part{}.tif'.format(num))+ ' 0\n')
            num += 1

