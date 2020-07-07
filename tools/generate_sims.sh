python2 tools/get_test_sim.py trained_models/r64-arcface-IJBC_RETINA/model 40000 final_predictions/r64-arcface-IJBC_RETINA/predictions.csv datasets/ECCV_TEST_ALIGNED/


python2 tools/get_test_sim.py trained_models/r100-arcface-IJBC_ORI/model 40000 final_predictions/r100-arcface-IJBC_ORI/predictions.csv datasets/ECCV_TEST_ORI/


python2 tools/get_test_sim.py trained_models/r100-arcface-ms1m/model 185000 final_predictions/r100-arcface-ms1m/predictions.csv datasets/ECCV_TEST_ALIGNED/


python2 tools/sim_combine.py final_predictions/r64-arcface-IJBC_RETINA/predictions.csv final_predictions/r100-arcface-IJBC_ORI/predictions.csv final_predictions/r100-arcface-ms1m/predictions.csv final_predictions/predictions.csv

