import re, fileinput, math
import numpy as np
import sys
from PIL import Image
import os
import random
import scipy.ndimage
import math
import cv2
from tqdm import tqdm

sys.path.append('./utils');
from evaluation_utils import *

data_path = 'KITTI/kitti_raw_data/'

# max_depth = 80;
# img_width = 621;
# img_height = 188;

# img_files = [];
# img_labels = [];

#get the list of rgb images
sequence = "0009"
fid = open(f'./utils/filenames/eigen_train_files_sync_{sequence}.txt', 'r')
img_lines = fid.readlines();
fid1 = open(f'./utils/filenames/eigen_train_pairs_sync_{sequence}.txt', 'w')

gt16bit_dir = f'KITTI/gt16bit_{sequence}/'

if not os.path.isdir(gt16bit_dir):
    os.makedirs(gt16bit_dir)

interpolate = False

for in_idx in tqdm(range(len(img_lines))):
    #print('Processing %d-th image ...' % in_idx)
    img_lines0 = img_lines[in_idx].split(' ')[0]
    index_f = str(in_idx+1);
    img_name = index_f.zfill(5) + '.png';

    #load image and depth
    gt_file, gt_calib, im_size, im_file, cams = read_file_data_new(img_lines[in_idx], data_path);
    camera_id = cams[0];
    
    depth_map = None

    if interpolate:
        depth, depth_interp = generate_depth_map(gt_calib[0], gt_file[0], im_size[0], camera_id, interpolate, True);
        depth_map = depth_interp
    else:
        depth = generate_depth_map(gt_calib[0], gt_file[0], im_size[0], camera_id, interpolate, True);
        depth_map = depth

    im_depth_16 = (depth_map * 100).astype(np.uint16);
    filename2 = os.path.join(gt16bit_dir, img_name);
    file_line = os.path.join('kitti_raw_data', img_lines0) + ' ' + os.path.join(f'gt16bit_{sequence}', img_name) + '\n'
    fid1.write(file_line)
    cv2.imwrite(filename2, im_depth_16)
fid.close()
fid1.close()
