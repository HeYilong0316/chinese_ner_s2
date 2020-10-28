# unzip data
unzip -d ../user_data/data ../data/round1_test.zip
unzip -d ../user_data/data ../data/round1_train.zip

# postprocess data
python data_process/postprocess_kfold.py

# zip
zip -r ../prediction_result/result.zip ../user_data/result