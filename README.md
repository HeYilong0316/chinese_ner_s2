# 运行环境

1. 操作系统
    - ubuntu 18.04
2. 深度学习框架及GPU驱动
    - python 3.7.3
    - pytorch 1.4.0
    - cudatoolkit 10.0.130
    - CUDA 10.0
    - cudnn 7.6.3  
3. 其他环境
    - python package 请详见 requirements.txt
    ```bash
    pip install -r requirements.txt
    ```
    
# 解决方法及算法

1. 预处理
    - 删除文本中无意义的字符，html标签，乱码等，空格以[unused1]代替
    - 对每一个说明书文本按两个空格，句子序号，句号，分号等分句之后，按照最长510个字符将分句后的结果重新组合成新的句子。
    - 以说明书文本的层级，将训练数据分为十折，其中每一折分别作为验证集，其余九折作为训练集，得到十组train.txt和dev.txt 
2. 模型
    - 本项目基于开源开源项目[transformers](https://github.com/huggingface/transformers)改写，在其基础上增加了下列结构：预训练多层动态加权融合，以attention的机制自动学习每一层表征的权重(下文以layer表示)，idcnn，lstm， crf, 多模型融合等
    - bert层，layer层，lstm/idcnn层，crf层均使用不同的学习率，从低到高依次递增。
    - 分将roberta后接lstm+crf，idcnn+crf，layer+lstm+crf，layer+idcnn+crf得到四种异构模型，并将每个模型在十折数据上分别训练，选取验证集上F1值最高的模型作为当前折的模型，最终得到40个模型，分别对测试集进行预测，得到40个预测结果。
    - 采用了重计算的方法，使得可以在2080ti(11G显存)上微调large版本的Roberta
3. 后处理
    - 对于每一个说明书文件的40个结果，以span等级(start, end, tag)进行投票，保留票数过半(>=20)的实体作为当前说明书的最终实体
4. 预训练模型后接结构对线上结果的影响(十折投票)

    | 结构 | F1 |
    | --- | --- |
    | +CRF(baseline) | 0.7715 |
    | +layer+CRF | 0.7740 |
    | +layer+lstm+crf | 0.7784 |
    | +layer+idcnn+crf | 0.7776 |
    | (+layer+idcnn+crf)+(+layer+lstm+crf)投票  | 0.7818 |
    | (+lstm+crf)+(+idcnn+crf)+(+layer+lstm+crf)+(+layer+idcnn+crf)投票  | 0.7824 |

5. 说明
    - 由于个人原因初赛阶段参赛时间较紧张，没有时间筛选模型，直接融合了所有模型，模型较多，运行时间较长，较难通过现场训练复现
    - 由于初赛所有模型较大(超过10G)，没有办法上传训练好的模型，仅上传初始化的开源WWM-Roberta模型(详见下文)
    - 复赛会根据要求，缩短运行时间，在规定时间内复现
    - 各模型的预测文件已放在相应的目录中，若想复现初赛结果可直接运行投票代码
        ```bash
        cd code
        sh vote.sh
        ```


# 预训练模型
1. 本项目使用的预训练模型是：[RoBERTa-wwm-ext-large, Chinese](https://github.com/ymcui/Chinese-BERT-wwm)
2. 将预训练模型解压后放入user_data/model/chinese-roberta-wwm-large-ext目录中，其文件名应修改为如下所示
>user_data/model/chinese-roberta-wwm-large-ext/
├── config.json
├── pytorch_model.bin
└── vocab.txt
3. 修改config.json中的内容为
    ```json
    {
      "attention_probs_dropout_prob": 0.1, 
      "directionality": "bidi", 
      "hidden_act": "gelu", 
      "hidden_dropout_prob": 0.1, 
      "hidden_size": 1024,
      "initializer_range": 0.02, 
      "intermediate_size": 4096, 
      "max_position_embeddings": 512, 
      "num_attention_heads": 16, 
      "num_hidden_layers": 24, 
      "pooler_fc_size": 768, 
      "pooler_num_attention_heads": 12, 
      "pooler_num_fc_layers": 3, 
      "pooler_size_per_head": 128, 
      "pooler_type": "first_token_transform", 
      "type_vocab_size": 2, 
      "vocab_size": 21128,
      "model_type": "bert"
    }
    ```
# 运行代码
1. 进入代码目录
```bash
cd code
```

2. 运行完整代码
```bash
sh main.sh
```

3. 运行投票代码(仅仅是将各模型的预测结果投票融合，为了在没有模型的情况下复现结果，第二步完整代码已包含了此步骤)
```bash
sh vote.sh
```
