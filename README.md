# Project for ECCV 2020 ChaLearn Looking at People Fair Face Recognition challenge 
change based insightface
------
## Requirements
1. python 2.7.12+
    - opencv
    - sklearn
    - numpy
    - scipy
2. mxnet >= 1.3.0
provided docker:  zhaixingzhaiyue/mxnetcu90-py2

## Instructions to reproduce the result of test phase with trained model

1. preprocess the test data 
     Download the preprocessed data from the url https://c-t.work/s/9afa03a0bbb74c, and put it in the datasets folder. Then run `tar xvf eccv_test_preprocessed.tar`

    If you want preprocess the test data yourself:

    Download retina face model(https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) to `facealign/RetinaFace/model`

    `cd facealign; sh ../tools/preprocess.sh` # you need to change the eccv_test_data to real path in ../tools/preprocess.sh. 
    Note, this is a little slow.
2. prepare the trained model
    Downlaod the trained model from https://c-t.work/s/9afa03a0bbb74c, and put it it the folder trained_models. Then run `tar xvf trained_models.tar`

3. **generate final predict file**

    `sh ./tools/generate_sims.sh`

    The final result will generate in the final_predictions directory with the name 'predictions.csv'


## Instructions to train

1. train the model with ms1m dataset
    ```sh
    cd recognition/
    sh ./scripts/train_ms1m.sh
    ```
    stop at 185000 iterations
2. train the model with aligned ijbc

    ```sh
    cd recognition/
    sh ./scripts/train_ijbc_aligned.sh
    ```
    stop at 40000 iterations
3. train the model with origin ijbc

    ```sh
    cd recognition/
    sh ./scripts/train_ijbc_ori.sh
    ```
    stop at 40000 iterations
