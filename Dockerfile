# Base Images
## 从天池基础镜像构建
FROM ner:v0.2

## 环境变量，用于判断是否在docker容器内运行
ENV AM_I_IN_A_DOCKER_CONTAINER Yes

## 数据层
COPY ./user_data/data /user_data/data
## 模型层
COPY ./user_data/model /user_data/model
## 代码层
COPY ./code /code

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /code

## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]