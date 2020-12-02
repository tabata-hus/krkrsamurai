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
cutSize = 10
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
            if(
                num%imageWidth==19 or
                num%imageWidth==20 or
                num%imageWidth==21 or
                num%imageWidth==22 or
                num%imageWidth==23 or
                num%imageWidth==24 or
                num%imageWidth==25 or
                num%imageWidth==26 or
                num%imageWidth==27 or
                num%imageWidth==28 or
                num%imageWidth==29 or
                num%imageWidth==30 or
                num%imageWidth==31 or
                num%imageWidth==32 or
                num%imageWidth==33 or
                num%imageWidth==34 or
                num%imageWidth==35 or
                num%imageWidth==36 or
                num%imageWidth==37 or
                num%imageWidth==38 or
                num%imageWidth==39 or
                num%imageWidth==40 or
                num%imageWidth==41 or
                num%imageWidth==42 or
                num%imageWidth==43 or
                num%imageWidth==44 or
                num%imageWidth==45 or
                num%imageWidth==46 or
                num%imageWidth==47 or
                num%imageWidth==48 or
                num%imageWidth==49


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
            cut_range = (0+i*misalignment,0+j*misalignment,cutSize+i*misalignment,cutSize+j*misalignment)
            im_crop = image[jj].crop(cut_range)
            im_crop.save('./test/cut_part{}.tif'.format(num))
            if(
                num%imageWidth==19 or
                num%imageWidth==20 or
                num%imageWidth==21 or
                num%imageWidth==22 or
                num%imageWidth==23 or
                num%imageWidth==24 or
                num%imageWidth==25 or
                num%imageWidth==26 or
                num%imageWidth==27 or
                num%imageWidth==28 or
                num%imageWidth==29 or
                num%imageWidth==30 or
                num%imageWidth==31 or
                num%imageWidth==32 or
                num%imageWidth==33 or
                num%imageWidth==34 or
                num%imageWidth==35 or
                num%imageWidth==36 or
                num%imageWidth==37 or
                num%imageWidth==38 or
                num%imageWidth==39 or
                num%imageWidth==40 or
                num%imageWidth==41 or
                num%imageWidth==42 or
                num%imageWidth==43 or
                num%imageWidth==44 or
                num%imageWidth==45 or
                num%imageWidth==46 or
                num%imageWidth==47 or
                num%imageWidth==48 or
                num%imageWidth==49
            ):
                file.writelines(('./test/cut_part{}.tif'.format(num))+ ' 1\n')
            else:
                file.writelines(('./test/cut_part{}.tif'.format(num))+ ' 0\n')
            num += 1

