# -*- coding: utf-8 -*-
"""

Created on Fri Mar 22 09:28:17 2019

@author: Amos

Reference:
    [1] Image denoising by sparse 3D transform-domain collaborative filtering
    [2] An Analysis and Implementation of the BM3D Image Denoising Method

"""

import os
import cv2
import time
import sys
import argparse
from scipy.fftpack import dct, idct
import numpy as np
from tqdm import tqdm
import argparse
from Transforms import *

# ==================================================================================================
#                                              Macros
# ==================================================================================================

# ==================================================================================================
#                                           Preprocessing
# ==================================================================================================

def AddNoise(Img, sigma):

    GuassNoise = np.random.normal(0, sigma, Img.shape)

    noisyImg = Img + GuassNoise

    return noisyImg


def Initialization(Img, BlockSize, Kaiser_Window_beta):

    InitImg = np.zeros(Img.shape, dtype=float)

    InitWeight = np.zeros(Img.shape, dtype=float)

    Window = np.matrix(np.kaiser(BlockSize, Kaiser_Window_beta))

    InitKaiser = np.array(Window.T * Window)

    return InitImg, InitWeight, InitKaiser


def SearchWindow(Img, RefPoint, BlockSize, WindowSize):

    if BlockSize >= WindowSize:
        print('Error: BlockSize is smaller than WindowSize.\n')

        exit()

    Margin = np.zeros((2, 2), dtype=int)

    Margin[0, 0] = max(0, RefPoint[0] + int((BlockSize - WindowSize) / 2))  # left-top x

    Margin[0, 1] = max(0, RefPoint[1] + int((BlockSize - WindowSize) / 2))  # left-top y

    Margin[1, 0] = Margin[0, 0] + WindowSize  # right-bottom x

    Margin[1, 1] = Margin[0, 1] + WindowSize  # right-bottom y

    if Margin[1, 0] >= Img.shape[0]:
        Margin[1, 0] = Img.shape[0] - 1

        Margin[0, 0] = Margin[1, 0] - WindowSize

    if Margin[1, 1] >= Img.shape[1]:
        Margin[1, 1] = Img.shape[1] - 1

        Margin[0, 1] = Margin[1, 1] - WindowSize

    return Margin


def dct2D(A):

    return dct(dct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2D(A):

    return idct(idct(A, axis=0, norm='ortho'), axis=1, norm='ortho')


def PreDCT(Img, BlockSize):

    BlockDCT_all = np.zeros((Img.shape[0] - BlockSize, Img.shape[1] - BlockSize, BlockSize, BlockSize), \
                            dtype=float)

    for i in range(BlockDCT_all.shape[0]):

        for j in range(BlockDCT_all.shape[1]):
            Block = Img[i:i + BlockSize, j:j + BlockSize]

            BlockDCT_all[i, j, :, :] = dct2D(Block.astype(np.float64))

    return BlockDCT_all


# ==================================================================================================
#                                         Basic estimate
# ==================================================================================================

def Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, ThreDist, MaxMatch, WindowSize, sigma, lamb2d):

    WindowLoc = SearchWindow(noisyImg, RefPoint, BlockSize, WindowSize)

    Block_Num_Searched = (WindowSize - BlockSize + 1) ** 2  # number of searched blocks

    BlockPos = np.zeros((Block_Num_Searched, 2), dtype=int)

    BlockGroup = np.zeros((Block_Num_Searched, BlockSize, BlockSize), dtype=float)

    Dist = np.zeros(Block_Num_Searched, dtype=float)

    RefDCT = BlockDCT_all[RefPoint[0], RefPoint[1], :, :]

    match_cnt = 0

    # Block searching and similarity (distance) computing

    for i in range(WindowSize - BlockSize + 1):

        for j in range(WindowSize - BlockSize + 1):

            SearchedDCT = BlockDCT_all[WindowLoc[0, 0] + i, WindowLoc[0, 1] + j, :, :]

            dist = ComputeDist(RefDCT, SearchedDCT, sigma, lamb2d)

            if dist < ThreDist:
                BlockPos[match_cnt, :] = [WindowLoc[0, 0] + i, WindowLoc[0, 1] + j]

                BlockGroup[match_cnt, :, :] = SearchedDCT

                Dist[match_cnt] = dist

                match_cnt += 1

    if match_cnt <= MaxMatch:

        # less than MaxMatch similar blocks founded, return similar blocks

        BlockPos = BlockPos[:match_cnt, :]

        BlockGroup = BlockGroup[:match_cnt, :, :]

    else:

        # more than MaxMatch similar blocks founded, return MaxMatch similarest blocks

        idx = np.argpartition(Dist[:match_cnt], MaxMatch)  # indices of MaxMatch smallest distances

        BlockPos = BlockPos[idx[:MaxMatch], :]

        BlockGroup = BlockGroup[idx[:MaxMatch], :]

    return BlockPos, BlockGroup


def ComputeDist(BlockDCT1, BlockDCT2, sigma, lamb2d):

    if BlockDCT1.shape != BlockDCT1.shape:

        print('ERROR: two DCT Blocks are not at the same shape in step1 computing distance.\n')

        sys.exit()

    elif BlockDCT1.shape[0] != BlockDCT1.shape[1]:

        print('ERROR: DCT Block is not square in step1 computing distance.\n')

        sys.exit()

    BlockSize = BlockDCT1.shape[0]

    if sigma >= 40:
        ThreValue = lamb2d * sigma

        BlockDCT1 = np.where(abs(BlockDCT1) < ThreValue, 0, BlockDCT1)

        BlockDCT2 = np.where(abs(BlockDCT2) < ThreValue, 0, BlockDCT2)

    return np.linalg.norm(BlockDCT1 - BlockDCT2) ** 2 / (BlockSize ** 2)


def Filtering(BlockGroup, sigma, lamb3d):

    ThreValue = lamb3d * sigma

    nonzero_cnt = 0

    # since 2D transform has been done, we do 1D transform, hard-thresholding and inverse 1D
    # transform, the inverse 2D transform is left in aggregation processing

    for i in range(BlockGroup.shape[1]):

        for j in range(BlockGroup.shape[2]):
            ThirdVector = dct(BlockGroup[:, i, j], norm='ortho')  # 1D DCT

            ThirdVector[abs(ThirdVector[:]) < ThreValue] = 0.

            nonzero_cnt += np.nonzero(ThirdVector)[0].size

            BlockGroup[:, i, j] = list(idct(ThirdVector, norm='ortho'))

    return BlockGroup, nonzero_cnt


def Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt, sigma):

    if nonzero_cnt < 1:

        BlockWeight = 1.0 * basicKaiser

    else:

        BlockWeight = (1. / (sigma ** 2 * nonzero_cnt)) * basicKaiser

    for i in range(BlockPos.shape[0]):
        basicImg[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1], \
        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] \
            += BlockWeight * idct2D(BlockGroup[i, :, :])

        basicWeight[BlockPos[i, 0]:BlockPos[i, 0] + BlockGroup.shape[1], \
        BlockPos[i, 1]:BlockPos[i, 1] + BlockGroup.shape[2]] += BlockWeight


def BM3D_Step1(noisyImg, config):

    # Config
    sigma = config.sigma
    BlockSize = config.BlockSize
    ThreDist = config.ThreDist
    MaxMatch = config.MaxMatch
    WindowSize = config.WindowSize
    spdup_factor = config.spdup_factor
    Kaiser_Window_beta = config.Kaiser_Window_beta
    lamb2d = config.lamb2d
    lamb3d = config.lamb3d
    transform_type = config.Transform_Type
    wavelet = config.Wavelet_Type
    # preprocessing
    basicImg, basicWeight, basicKaiser = Initialization(noisyImg, BlockSize, Kaiser_Window_beta)

    BlockDCT_all = PreDCT(noisyImg, BlockSize)


    # block-wise estimate with speed-up factor
    for i in tqdm(range(int((noisyImg.shape[0] - BlockSize) / spdup_factor) + 2)):

        for j in range(int((noisyImg.shape[1] - BlockSize) / spdup_factor) + 2):
            RefPoint = [min(spdup_factor * i, noisyImg.shape[0] - BlockSize - 1), \
                        min(spdup_factor * j, noisyImg.shape[1] - BlockSize - 1)]
            BlockPos, BlockGroup = Grouping(noisyImg, RefPoint, BlockDCT_all, BlockSize, \
                                                  ThreDist, MaxMatch, WindowSize, sigma, lamb2d)
            if transform_type == "DCT_Wavelet":
                BlockGroup, nonzero_cnt = dct_dwt_transform(BlockGroup, sigma, lamb3d, wavelet)
            elif transform_type == "Wavelet":
                BlockGroup, nonzero_cnt = full_wavelet_3d_transform(BlockGroup, sigma, lamb3d, wavelet)
            elif transform_type == "3D_DCT":
                BlockGroup, nonzero_cnt = Filtering(BlockGroup, sigma, lamb3d)
            else:
                raise NotImplementedError

            Aggregation(BlockGroup, BlockPos, basicImg, basicWeight, basicKaiser, nonzero_cnt, sigma)
    basicWeight = np.where(basicWeight == 0, 1, basicWeight)
    basicImg[:, :] /= basicWeight[:, :]

    return basicImg