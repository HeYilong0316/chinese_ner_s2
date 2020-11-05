docker build -t registry.cn-shenzhen.aliyuncs.com/heyilong_tianchi/chinese_medical_ner:large_layer_words_lstm .
docker rmi -f $(docker images | grep "none" | awk '{print $3}')