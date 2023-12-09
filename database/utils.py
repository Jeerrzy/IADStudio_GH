#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : Jeerrzy
# @File     : utils.py
# @Time     : 2023/11/21 15:38
# @Project  : GH_Industry_Defect


import time
import os
import cv2
import math
import numpy as np
import imgaug.augmenters as iaa


def imageSlice(imagePath, outDir, sliceHeight=256, sliceWidth=256, overlap=0.2, non_zero_frac_thresh=0.8, verbose=False):
    """
    将图像按给定尺度切分并保存到输出路径中
    :param imagePath: 输入图片路径
    :param outDir: 输出图片目录
    :param sliceHeight: 切分的图像高度
    :param sliceWidth: 切分的图像宽度
    :param overlap: 切分的重叠度
    :param non_zero_frac_thresh: 非零像素占据全部的图像的阈值
    :param verbose: 是否在控制台输出程序信息
    :return: None
    """
    t0 = time.time()
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    baseName = os.path.basename(imagePath)
    name, ext = os.path.splitext(baseName)
    imageData = cv2.imread(imagePath)
    imageHeight, imageWidth = imageData.shape[:2]
    # if slice sizes are large than image, pad the edges
    pad = max(0, sliceHeight - imageHeight, sliceWidth - imageWidth)
    # pad the edge of the image with black pixels
    if pad > 0:
        borderColor = (0, 0, 0)
        imageData = cv2.copyMakeBorder(imageData, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=borderColor)
    n_slice = 0
    sliceSize = sliceHeight * sliceWidth
    dx = int((1. - overlap) * sliceWidth)
    dy = int((1. - overlap) * sliceHeight)
    for y0 in range(0, imageHeight, dy):  # sliceHeight:
        for x0 in range(0, imageWidth, dx):  # sliceWidth:
            # make sure we don't have a tiny image on the edge
            y = imageHeight - sliceHeight if y0 + sliceHeight > imageHeight else y0
            x = imageWidth - sliceWidth if x0 + sliceWidth > imageWidth else x0
            # cut image
            slice = imageData[y:y + sliceHeight, x:x + sliceWidth]
            # skip if image is mostly empty
            ret, thresh = cv2.threshold(cv2.cvtColor(slice, cv2.COLOR_BGR2GRAY), 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh)
            non_zero_frac = float(non_zero_counts) / sliceSize
            if non_zero_frac >= non_zero_frac_thresh:
                n_slice += 1
                # save slice
                savePath = os.path.join(outDir, f'{name}_{n_slice}_{y}_{x}_{sliceHeight}_{sliceWidth}_{pad}_{imageHeight}_{imageWidth}{ext}')
                cv2.imwrite(savePath, slice)
                if verbose:
                    print(f'save {n_slice}th slice to {savePath}')
    cost_time = round(time.time() - t0, 2)
    if verbose:
        print(f'{imagePath} to {n_slice} slice.')
        print(f'imageHeight:{imageHeight}, imageWidth:{imageWidth}')
        print(f'sliceHeight:{sliceHeight}, sliceWidth:{sliceWidth}')
        print(f'Time to slice:{cost_time}s, outDir:{outDir}')


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


def rand_perlin_2d_np(
    shape, res, fade=lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: cv2.resize(
        np.repeat(
            np.repeat(
                gradients[slice1[0] : slice1[1], slice2[0] : slice2[1]], d[0], axis=0
            ),
            d[1],
            axis=1,
        ),
        dsize=(shape[1], shape[0]),
    )
    dot = lambda grad, shift: (
        np.stack(
            (
                grid[: shape[0], : shape[1], 0] + shift[0],
                grid[: shape[0], : shape[1], 1] + shift[1],
            ),
            axis=-1,
        )
        * grad[: shape[0], : shape[1]]
    ).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[: shape[0], : shape[1]])
    return math.sqrt(2) * lerp_np(
        lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]
    )


rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


def perlin_noise(image, dtd_image, aug_prob=1.0):
    image = np.array(image, dtype=np.float32)
    dtd_image = np.array(dtd_image, dtype=np.float32)
    shape = image.shape[:2]
    min_perlin_scale, max_perlin_scale = 0, 6
    t_x = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    t_y = torch.randint(min_perlin_scale, max_perlin_scale, (1,)).numpy()[0]
    perlin_scalex, perlin_scaley = 2**t_x, 2**t_y

    perlin_noise = rand_perlin_2d_np(shape, (perlin_scalex, perlin_scaley))

    perlin_noise = rot(images=perlin_noise)
    perlin_noise = np.expand_dims(perlin_noise, axis=2)
    threshold = 0.5
    perlin_thr = np.where(
        perlin_noise > threshold,
        np.ones_like(perlin_noise),
        np.zeros_like(perlin_noise),
    )

    img_thr = dtd_image * perlin_thr / 255.0
    image = image / 255.0

    beta = torch.rand(1).numpy()[0] * 0.8
    image_aug = (
        image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)
    )
    image_aug = image_aug.astype(np.float32)

    no_anomaly = torch.rand(1).numpy()[0]

    if no_anomaly > aug_prob:
        return image, np.zeros_like(perlin_thr)
    else:
        msk = (perlin_thr).astype(np.float32)
        msk = msk.transpose(2, 0, 1)

        return image_aug, msk