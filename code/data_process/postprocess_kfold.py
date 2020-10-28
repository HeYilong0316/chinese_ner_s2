import sys
print(sys.path)

K_FOLD = 10

import pandas as pd
import os
import shutil
import json
from seqeval.metrics.sequence_labeling import get_entities
import zipfile

os.chdir(os.getcwd())

class PostProcess:
    def __init__(self, predict_file, txt_file_dir):
        
        predict_datas = []
        predict_data = []
        predict_example = []

        with open(predict_file, "r", encoding="utf8") as r:
            for line in r:
                if line.strip() == "":
                    labels = [e[-1] for e in predict_example]
                    predict_example = [[e[0], e[1], e[2], l] for e, l in zip(predict_example, labels)]
                    predict_data.extend(predict_example) 
                    predict_example = []
                    continue    
                else:
                    splits = line.strip().split(" ")
                    if len(splits) == 7:
                        token, file_id, poses, _, label, new_token, _ = splits
                        if new_token == "[UNK]":
                            # print(token)
                            pass
                        else:
                            assert token.lower() == new_token, [token, new_token]
                    else:
                        token = " "
                        file_id, poses, label, new_token, _ = splits
                        assert new_token == "[unused1]", [new_token]

                    if predict_data and file_id != predict_data[0][1]:
                        predict_datas.append(predict_data)
                        predict_data = []  
                    predict_example.append([token, file_id, poses, label])
            
        if predict_example:
            predict_data.extend(predict_example) 
        
        if predict_data:
            predict_datas.append(predict_data)
        self.predict_datas = predict_datas
        self.txt_file_dir = txt_file_dir

    
    def conll2brat(self): 
        submit_dict = {}
        for predict_data in self.predict_datas:
            file_id = predict_data[0][1]
            with open(os.path.join(self.txt_file_dir, f"{file_id}.txt"), "r", encoding="utf8") as r:
                text = r.read()

            labels = [d[-1] for d in predict_data]
            entities = get_entities(labels)

            for i in range(len(entities)):
                entities[i] = list(entities[i])
                start, end = entities[i][1], entities[i][2]
                entities[i][1] = int(predict_data[start][2])
                entities[i][2] = int(predict_data[end][2])
                entities[i] = tuple(entities[i])
            submit_dict[file_id] = entities
        return submit_dict


def save(file_id, entities, path, txt_file_dir):
    with open(os.path.join(txt_file_dir, f"{file_id}.txt"), "r", encoding="utf8") as r:
        text = r.read()
    to_save = ""
    entities = sorted(entities, key=lambda x: (x[1], x[2]))
    for i, (label, real_start, real_end) in enumerate(entities):
        real_end += 1
        to_save += f"T{i}\t{label} {real_start} {real_end}\t{text[real_start:real_end]}\n"
    with open(os.path.join(path, f"{file_id}.ann"), "w", encoding="utf8") as w:
        w.write(to_save)

def vote(submit_k_fold, n=None):
    '''投票'''
    import numpy as np
    if n is None:
        # 过半数票
        n = np.ceil(len(submit_k_fold) / 2)
        print(f"票数过半: {n}")

    submit = []
    all_file_id = set()
    for k_fold_examples in submit_k_fold:
        all_file_id.update(k_fold_examples.keys())

    all_file_id = sorted(list(all_file_id))

    for file_id in all_file_id:
        entity_count_dict = {}
        for i, one_fold_examples in enumerate(submit_k_fold):
            one_file_entities = one_fold_examples.get(file_id, [])
            for entity in one_file_entities:
                entity_count_dict[entity] = entity_count_dict.get(entity, 0) + 1
        entities = [e for e, c in entity_count_dict.items() if c >= n]
        submit.append((file_id, entities))
    return submit

def main():
    submit_k_fold = []
    txt_file_dir = "/tcdata/juesai"
    for i in range(K_FOLD):
        # predict_file = f"../user_data/output_idcnn_crf/{i}_fold/test_predictions.txt"
        # submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
        # submit_k_fold.append(submit_one_fold)
        
        predict_file = f"../user_data/output/output_layer_lstm_crf/{i}_fold/test_predictions.txt"
        submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
        submit_k_fold.append(submit_one_fold)
        
        predict_file = f"../user_data/output/output_layer_idcnn_crf/{i}_fold/test_predictions.txt"     
        submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
        submit_k_fold.append(submit_one_fold)

        # predict_file = f"../user_data/output/output_layer_output_crf/{i}_fold/test_predictions.txt"     
        # submit_one_fold = PostProcess(predict_file, txt_file_dir).conll2brat()
        # submit_k_fold.append(submit_one_fold)

    submit = vote(submit_k_fold)


    save_dir = "./result"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) # 能删除该文件夹和文件夹下所有文件
    os.mkdir(save_dir)

    for file_id, entities in submit:
        save(file_id, entities, save_dir, txt_file_dir)
    
    # 没有预测出结果的文件留空
    from glob import glob
    test_ann_list = glob("./result/*.ann")
    test_ann_list = [os.path.basename(t).split(".")[0] for t in test_ann_list]
    test_txt_list = glob("/tcdata/juesai/*.txt")
    test_txt_list = [os.path.basename(t).split(".")[0] for t in test_txt_list]

    none_pred_list = list(set(test_txt_list)-set(test_ann_list))
    print(none_pred_list)

    for file_id in none_pred_list:
        with open(os.path.join("./result", f"{file_id}.ann"), "w", encoding="utf8") as w:
            w.write("")



    

if __name__ == "__main__":
    main()


            
                








