cp -f configs/ms1m_config.py config.py 
export dropout=1 
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python2 -u train.py --network r100 --loss arcface --dataset ms1m