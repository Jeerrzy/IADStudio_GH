#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    : demo
# @Date    : 2023/11/19 0:56
# @Author  : Jeerrzy


import os
import cv2
import json


TileDatasetRoot = 'E:/tile_round1_train_20201231'
TileDatasetFileRoot = os.path.join(TileDatasetRoot, 'train_imgs')
TileDatasetFileInfoPath = os.path.join(TileDatasetRoot, 'train_annos.json')


if __name__ == "__main__":
    idx = 1
    with open(TileDatasetFileInfoPath, 'r') as f:
        datasetInfo = json.load(f)
    fileInfo = datasetInfo[idx]
    print(fileInfo)
    imageData = cv2.imread(os.path.join(TileDatasetFileRoot, fileInfo['name']))
    imageHeight, imageWidth = fileInfo['image_height'], fileInfo['image_width']
    x1y1x2y2Bbox = list(map(lambda x: int(x), fileInfo['bbox']))
    (x1, y1, x2, y2) = x1y1x2y2Bbox
    cv2.rectangle(imageData, (x1, y1), (x2, y2), (0, 255, 0), 1)
    imageDataResized = cv2.resize(imageData, (int(imageWidth*0.2), int(imageHeight*0.2)))
    cv2.imshow('demo', imageDataResized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

