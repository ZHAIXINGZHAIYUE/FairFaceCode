
eccv_test_data='' # put origin test_data dir here, the aligned imgs will appear in  ../datasets/ECCV_TEST_ALIGNED and ../datasets/ECCV_TEST_ORI
python2 facealign/detect_and_align.py ${eccv_test_data} ../datasets/ECCV_TEST_ALIGNED ../datasets/test_template/test_list.txt

python2 facealign/resize.py ${eccv_test_data} ../datasets/ECCV_TEST_ORI ../datasets/test_template/test_list.txt