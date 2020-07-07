#coding=utf-8
import cv2
import sys
import numpy as np
import datetime
import os
import glob
from skimage import transform as trans
from retinaface import RetinaFace


if __name__ == '__main__':
    img_list_path = sys.argv[3]
    root_dir = sys.argv[1]
    out_dir =  sys.argv[2]
    img_list = [os.path.basename(it.strip()) for it in open(img_list_path, 'r').readlines()]
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for path in img_list:
        src_path = os.path.join(root_dir, path)
        dst_path = os.path.join(out_dir,path)
        img = cv2.imread(src_path)
        img_dst = cv2.resize(warped,(96, 112))
        cv2.imwrite(dst_path, img_dst)


    

