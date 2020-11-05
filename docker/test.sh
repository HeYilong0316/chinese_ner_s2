docker stop ner
docker rm ner
nvidia-docker run --name ner \
-v /home/heyilong/codes/chinese_medical_ner_submit/project/data/chusai_xuanshou:/tcdata/juesai \
-v /home/heyilong/codes/chinese_medical_ner_submit/project/user_data/output:/user_data/output \
registry.cn-shenzhen.aliyuncs.com/heyilong_tianchi/chinese_medical_ner:large_layer_words_lstm sh run.sh