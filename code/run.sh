# # unzip data
# unzip -d ../user_data ../data/round1_test.zip
# unzip -d ../user_data ../data/round1_train.zip

# prepeocess data
python ./data_process/preprocess.py k-fold predict

# train 
# sh script/train/run_k_fold.sh

# # predict
sh script/predict/run_k_fold.sh

# postprocess data
python data_process/postprocess_kfold.py 

# zip
rm ./result.zip
zip -r result.zip result
rm -r result