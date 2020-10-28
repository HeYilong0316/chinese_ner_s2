docker build -t registry.cn-shenzhen.aliyuncs.com/heyilong_tianchi/chinese_medical_ner:lstm_idcnn .
docker rmi $(docker images | grep "none" | awk '{print $3}')