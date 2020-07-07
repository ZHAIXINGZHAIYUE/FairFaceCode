# coding=utf-8
# author:zhangkun
from __future__ import division
from __future__ import print_function

import sys
import os
import cv2
import numpy as np
import mxnet as mx
import sklearn.preprocessing
import readline
import scipy.io as scio
import pdb
from tqdm import tqdm


predictions_template = 'datasets/test_template/predictions.csv' # 模板
img_list = 'datasets/test_template/test_list.txt' # 原始图像列表



def get_model(prefix, epoch, output='fc5'):
    """
    获取模型
    """
    gpu = os.getenv('gpu')
    if gpu is None:
        gpu = int(1)
    gpu = int(gpu)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[output+'_output']
    model = mx.mod.Module(symbol=sym, context=[mx.gpu(gpu)], label_names=None)
    model.bind(data_shapes=[('data', (1, 1, 112, 96))])
    model.set_params(arg_params, aux_params)
    return model


def main():
    prefix = sys.argv[1]
    iter_num = int(sys.argv[2])
    out = sys.argv[3]
    img_dir = sys.argv[4]
    model = get_model(prefix, iter_num)

    with open(img_list, 'r') as fin:
        features_dict = {}
        files = fin.readlines()
        for i, file in tqdm(enumerate(files)):
            p = os.path.join(img_dir, file.strip())
            img = cv2.imread(p, 0)
            img_nd = mx.nd.array(img).astype(np.float32)
            img1 = img_nd.expand_dims(axis=0).expand_dims(axis=0)
            db = mx.io.DataBatch((img1,))
            model.forward(db, is_train=False)
            features = model.get_outputs()[0].asnumpy()
            features = features.reshape((1, -1))
            normalized_features = sklearn.preprocessing.normalize(
                features, axis=1)
            features_dict[os.path.basename(p).split(
                '.')[0]] = normalized_features
    try:
        os.makedirs(os.path.dirname(out))
    except:
        pass

    with open(predictions_template, 'r') as fin, open(out, 'w') as fout:
        failed_pair = 0
        for i, line in tqdm(enumerate(fin.readlines())):
            if i == 0:
                fout.write(line)
            else:
                id1, id2, score = line.strip().split(',')
                if id1 in features_dict and id2 in features_dict:
                    score = np.matmul(features_dict[id1], features_dict[id2].T)[0][0]
                else:
                    score = 0
                    failed_pair += 1
                fout.write(id1+','+id2+','+str(score)+'\n')
    print('failed_pair ', failed_pair)


if __name__ == '__main__':
    if len(sys.argv) !=5:
        print('input: model_prefix epoch_num out_file img_dir')
    else:
        main()
