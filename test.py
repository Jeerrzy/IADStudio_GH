#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : test.py
# @Time     : 2023/11/21 16:48
# @Project  : GH_Industry_Defect


from database import *


def slice_test():
    imageSlice(
        imagePath='./dataset_2023_11_17/images/BG.jpg',
        outDir='./slice_test',
        sliceWidth=640,
        sliceHeight=640,
        verbose=True
    )


if __name__ == "__main__":
    slice_test()

