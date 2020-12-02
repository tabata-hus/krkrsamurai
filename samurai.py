from PIL import Image
import numpy as np
import cv2
image = [
    Image.open('./before/D7_d3_t1_PSFdeconvoluted_2020919_5.02ms.tif'),
    Image.open('./before/D7_d3_t1_PSFdeconvoluted_2020919_10.4ms.tif'),
    Image.open('./before/D7_d3_t1_PSFdeconvoluted_2020919_50.2ms.tif'),
    Image.open('./before/D7_d3_t1_PSFdeconvoluted_2020919_20.1ms.tif'),
    Image.open('./before/D7_d3_t2_PSFdeconvoluted_2020926_5.02ms.tif'),
    Image.open('./before/D7_d3_t2_PSFdeconvoluted_2020926_10.4ms.tif'),
    Image.open('./before/D7_d3_t2_PSFdeconvoluted_2020926_20.1ms.tif'),
    Image.open('./before/D7_d3_t2_PSFdeconvoluted_2020926_50.2ms.tif'),
    Image.open('./before/D7_d3_t3_PSFdeconvoluted_2020919_5.02ms.tif'),
    Image.open('./before/D7_d3_t3_PSFdeconvoluted_2020919_10.4ms.tif'),
    ]

path = './after/cut_set.txt'
cutSize = 5
imageWidth = 350 / cutSize
imageHeight = 400/ cutSize
file = open(path,'w')

num = 0
for jj in range(len(image)):
    for j in range(int(imageHeight)):
        for i in range(int(imageWidth)):
            cut_range = (0+i*cutSize,0+j*cutSize,cutSize+i*cutSize,cutSize+j*cutSize)
            im_crop = image[jj].crop(cut_range)
            im_crop.save('./after/cut_part{}.tif'.format(num))
            if(
               num%int(imageWidth)==16
            or num%int(imageWidth)==17
            or num%int(imageWidth)==18
            or num%int(imageWidth)==19
            or num%int(imageWidth)==20
            or num%int(imageWidth)==21
            or num%int(imageWidth)==22
            or num%int(imageWidth)==23
            or num%int(imageWidth)==24
            or num%int(imageWidth)==25
            or num%int(imageWidth)==26
            or num%int(imageWidth)==27
            or num%int(imageWidth)==28
            or num%int(imageWidth)==29
            or num%int(imageWidth)==30
            or num%int(imageWidth)==31
            or num%int(imageWidth)==32
            or num%int(imageWidth)==33
            or num%int(imageWidth)==34
            or num%int(imageWidth)==35
            or num%int(imageWidth)==36
            or num%int(imageWidth)==37
            or num%int(imageWidth)==38
            or num%int(imageWidth)==39
            or num%int(imageWidth)==40
            or num%int(imageWidth)==41
            or num%int(imageWidth)==42
            or num%int(imageWidth)==43
            or num%int(imageWidth)==44
            ):
                file.writelines(('./after/cut_part{}.tif'.format(num))+ ' 1\n')
            else:
                file.writelines(('./after/cut_part{}.tif'.format(num))+ ' 0\n')
            num += 1

image = [
    Image.open('./before/D7_d3_t3_PSFdeconvoluted_2020919_20.1ms.tif'),
    Image.open('./before/D7_d3_t3_PSFdeconvoluted_2020919_50.2ms.tif'),
    ]

path = './test/cut_set.txt'
file = open(path,'w')

num = 0
for jj in range(len(image)):
    for j in range(int(imageHeight)):
        for i in range((int(imageWidth))):
            cut_range = (0+i*cutSize,0+j*cutSize,cutSize+i*cutSize,cutSize+j*cutSize)
            im_crop = image[jj].crop(cut_range)
            im_crop.save('./test/cut_part{}.tif'.format(num))
            if(
               num%int(imageWidth)==16
            or num%int(imageWidth)==17
            or num%int(imageWidth)==18
            or num%int(imageWidth)==19
            or num%int(imageWidth)==20
            or num%int(imageWidth)==21
            or num%int(imageWidth)==22
            or num%int(imageWidth)==23
            or num%int(imageWidth)==24
            or num%int(imageWidth)==25
            or num%int(imageWidth)==26
            or num%int(imageWidth)==27
            or num%int(imageWidth)==28
            or num%int(imageWidth)==29
            or num%int(imageWidth)==30
            or num%int(imageWidth)==31
            or num%int(imageWidth)==32
            or num%int(imageWidth)==33
            or num%int(imageWidth)==34
            or num%int(imageWidth)==35
            or num%int(imageWidth)==36
            or num%int(imageWidth)==37
            or num%int(imageWidth)==38
            or num%int(imageWidth)==39
            or num%int(imageWidth)==40
            or num%int(imageWidth)==41
            or num%int(imageWidth)==42
            or num%int(imageWidth)==43
            or num%int(imageWidth)==44
            ):
                file.writelines(('./test/cut_part{}.tif'.format(num))+ ' 1\n')
            else:
                file.writelines(('./test/cut_part{}.tif'.format(num))+ ' 0\n')
            num += 1

