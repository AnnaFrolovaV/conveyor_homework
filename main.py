import debayering
from debayering import menon

import cv2 as cv
import numpy as ppool

import bm3d

from skimage import color
import cv2

import numpy as np
import matplotlib.pyplot as plt
from colour import sRGB_to_XYZ, XYZ_to_sRGB, RGB_to_XYZ
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd

from sklearn.linear_model import LinearRegression 

from seminars.practice_03_CST.utils.utils import RootPolynomialFeatures, RobustScalableRationalFeatures
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor, LinearRegression
from sklearn.pipeline import Pipeline

from tqdm.notebook import tqdm
import pickle
import argparse

def white_patch(img):
    # Convert the input image to chromaticity representation
    img_chroma = color.rgb2hsv(img)

    # Find the pixel with the highest chromaticity value in each channel
    max_chroma = np.amax(img_chroma, axis=(0, 1))

    return max_chroma

def linearize(img, black_lvl=2048, saturation_lvl=2**14-1):
    """
    :param saturation_lvl: 2**14-1 is a common value. Not all images
                           have the same value.
    """
    return (img - black_lvl)/(saturation_lvl - black_lvl)

cam2rgb = np.array([
        1.8795, -1.0326, 0.1531,
        -0.2198, 1.7153, -0.4955,
        0.0069, -0.5150, 1.5081,]).reshape((3, 3))

def max_rgb(image):
    """
    Applies the Max-RGB algorithm to an input RGB image.

    Args:
        image: numpy array of shape (height, width, 3) representing the input image

    Returns:
        corrected_image: numpy array of shape (3,) representing the RGB correction vector
    """

    # Compute the maximum value for each color channel of the input image
    max_values = np.max(np.max(image, axis=0), axis=0)

    return max_values


class MyDataset(Dataset):
    def __init__(self, df, ret_data):
        self.df = df
        self.ret_data = ret_data
#         self.bayer = Bayer()

    def __getitem__(self, idx):
        GT_name = self.df['gt_name'].loc[idx]
        SAMPLE_name = self.df['sample_name'].loc[idx]
        
        gt = np.load(GT_name, allow_pickle=True)
        sample = np.load(SAMPLE_name, allow_pickle=True)
        
        gt_xyz  = gt.item().get('xyz')
        gt_cmfs = gt.item().get('cmfs')
#         sRGB_gt = XYZ_to_sRGB(gt_xyz)

        sample_img   = sample.item().get('image')
        sample_cmfs  = sample.item().get('cmfs')
        sample_light = sample.item().get('light')
        sample_bayer = sample.item().get('bayer')
        sample_mean  = sample.item().get('mean')
        sample_sigma = sample.item().get('sigma')
        sample_xyz = sample.item().get('xyz')
#         if self.ret_data == 'img':
#             return sample_img, gt_xyz

        return sample_img, sample_bayer, sample_sigma, sample_light, sample_xyz, gt_xyz
    
    def __len__(self):
        return len(self.df)
    
def conveer(s_img, s_bayer, s_sigma, gt):
    rgb = menon.bayer2rgb(s_img, pattern='GBRG') # debayering
    norm = ppool.zeros((800,800))
    norm_im = cv.normalize(rgb,  norm, 0, 1, cv.NORM_MINMAX) # normalizing
    bm3d_img = bm3d.bm3d(norm_im, sigma_psd=s_sigma/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING) # denoising
    image = linearize(bm3d_img, np.min(bm3d_img), np.max(bm3d_img))
    illum = max_rgb(image)
    image = image/illum
    image = np.dot(image, cam2rgb.T)
    return bm3d_img, image

def res_type(image):
    return cv2.cvtColor((XYZ_to_sRGB(image) * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def process_images(input_path: str, output_path: str) -> None:
    test_im_list = os.listdir(input_path)

    test_gt = []
    test_sample = []
    for im_name in test_im_list:
        if im_name[-6:] == 'gt.npy':
            test_gt += ['./test_output/' + im_name]
            test_sample += ['./test_output/' + im_name[:-6] + 'sample.npy']

    test_data = pd.DataFrame()
    test_data['gt_name'] = np.array(test_gt)
    test_data['sample_name'] = np.array(test_sample)
    test_dataset = MyDataset(test_data, 'img')
    
    with open('model', 'rb') as file:
        model = pickle.load(file)

    for i in tqdm(range(len(test_dataset))):
        sample_img, sample_bayer, sample_sigma, sample_light, sample_xyz, gt_xyz = test_dataset[i]
        bm3d_img, image = conveer(sample_img, sample_bayer, sample_sigma, gt_xyz)
        pred = model.predict(image.reshape(-1, 3)).reshape(512, 512, 3)
        res = bm3d.bm3d(pred, sigma_psd=sample_sigma/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
        cv2.imwrite(f"./output_path/gt/{i}.png", res_type(gt_xyz))
        cv2.imwrite(f"./output_path/preds/{i}.png", res_type(res))


def parse_args() -> argparse.Namespace:
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate the quality of the predicted image.')
    parser.add_argument(
        'input_path',
        type=str,
        help='The path to the input images.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='The path to the output images.'
    )
    return parser.parse_args()


def main():
    """The main function."""
    args = parse_args()
    process_images(args.input_path, args.output_path)


if __name__ == '__main__':
    main()
