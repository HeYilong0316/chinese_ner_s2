# # unzip data
# unzip -d ../user_data/data ../data/round1_test.zip
# unzip -d ../user_data/data ../data/round1_train.zip

# prepeocess data
python ./data_process/preprocess.py 10 510 predict

# train 
sh script/run.sh

# postprocess data
python data_process/postprocess_kfold.py 

# zip
rm ./result.zip
zip -q -r result.zip result
rm -r result
echo "end"