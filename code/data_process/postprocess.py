import sys
sys.path.append("/home/heyilong/codes/chinese_medical_ner/")
print(sys.path)


import pandas as pd
import os
import shutil
import json
from seqeval.metrics.sequence_labeling import get_entities
from ruler.dict_extractor import dict_extractor

os.chdir(os.getcwd())

class PostProcess:
    def __init__(self, predict_file, txt_file_dir):
        
        predict_datas = []
        predict_data = []
        predict_example = []
        ori_labels = []
        # with open("./datas/conll/text_label.json", "r", encoding="utf8") as r:
        #     text_label_list = json.load(r)
        text_label_map = {}
        text_label_list = []
        for t in text_label_list:
            text = t["text"]
            label = t["label"]
            label_list = ["O"] * len(text)
            for tag, start, end, mention in label:
                label_list[start] = f"B-{tag}"
                for i in range(start+1, end+1):
                    label_list[i] = f"I-{tag}"
            text_label_map[text] = label_list

        conflict_case = []
        with open(predict_file, "r", encoding="utf8") as r:
            for line in r:
                if line.strip() == "":
                    text = "".join([e[0] for e in predict_example])
                    labels = [e[-1] for e in predict_example]
                    repair_labels = [e[-2] for e in predict_example]             
                    # if text in text_label_map:
                    #     assert text_label_map[text] == repair_labels, [text, text_label_map[text], repair_labels]
                    #     if text_label_map[text] != labels:
                    #         conflict_case.append(
                    #             list(zip(text, text_label_map[text], labels))
                    #         )
                    #     labels = text_label_map[text]
                    predict_example = [[e[0], e[1], e[2], l] for e, l in zip(predict_example, labels)]
                    predict_data.extend(predict_example) 
                    predict_example = []
                    continue    
                else:
                    splits = line.strip().split(" ")
                    if len(splits) == 7:
                        token, file_id, poses, repair_label, label, new_token, _ = splits
                        if new_token == "[UNK]":
                            print(token)
                        else:
                            assert token.lower() == new_token, [token, new_token]
                    else:
                        token = " "
                        repair_label = "O"
                        file_id, poses, label, new_token, _ = splits
                        assert new_token == "[unused1]", [new_token]

                    if predict_data and file_id != predict_data[0][1]:
                        predict_datas.append(predict_data)
                        predict_data = []  
                    predict_example.append([token, file_id, poses, repair_label, label])
        with open("model_ckpt/conflict_case.json", "w", encoding="utf8") as w:
            json.dump(conflict_case, w, ensure_ascii=False, indent=2)
        
        if predict_example:
            labels = [e[-1] for e in predict_example]
            ori_labels = [e[-2] for e in predict_example]
            if any([l != "O" for l in ori_labels]):
                labels = ori_labels
            predict_example = [[e[0], e[1], e[2], l] for e, l in zip(predict_example, labels)]
            predict_data.extend(predict_example) 
        
        if predict_data:
            predict_datas.append(predict_data)
        self.predict_datas = predict_datas
        self.txt_file_dir = txt_file_dir

        self.dict_extractor = dict_extractor

    def get_dict_entities(self, file_id):
        text_file = os.path.join("datas/brat/chusai_xuanshou", f"{file_id}.txt")
        with open(text_file, "r", encoding="utf8") as r:
            text = r.read()
        entities = self.dict_extractor.extract(text)
        return entities

    def recorrect_by_dict(self, model_entities, dict_entities, file_id):
        # model_entities = sorted(model_entities, key=lambda x: (x[1], x[2]))
        # dict_entities = sorted(dict_entities, key=lambda x: (x[1], x[2]))

        def span_is_over(ent1, ent2):
            ent1_index = set(range(ent1[1], ent1[2]+1))
            ent2_index = set(range(ent2[1], ent2[2]+1))
            overlap = ent1_index & ent2_index
            if overlap:
                # 有重叠
                return True
            return False

        entities = dict_entities
        for model_ent in model_entities:
            for dict_ent in dict_entities:
                if span_is_over(model_ent, dict_ent):
                    break
            else:
                entities.append(model_ent)


        entities = list(set(entities))
        entities = sorted(entities, key=lambda x: (x[1], x[2]))

        return entities

    
    def conll2brat(self):
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
                # entities[i][2] += 1

            # # 使用字典抽取实体
            # dict_entities = self.get_dict_entities(file_id)
            # # 使用字典校正结果
            # entities = self.recorrect_by_dict(entities, dict_entities, file_id)
            # # entities = dict_entities

            to_save = ""
            for i, (label, real_start, real_end) in enumerate(entities):
                real_end += 1
                to_save += f"T{i}\t{label} {real_start} {real_end}\t{text[real_start:real_end]}\n"
            with open(f"datas/submit/{file_id}.ann", "w", encoding="utf8") as w:
                w.write(to_save)

def main():
    # predict_file = "model_ckpt/test_predictions.txt"
    # predict_file = "model_ckpt/test_predictions.txt"
    predict_file = "output/test_predictions.txt"
    # predict_file = "submited_results/test_09281349_7571_cl/test_predictions.txt"
    # predict_file = "output_cur_sec/test_predictions.txt"
    txt_file_dir = "datas/brat/chusai_xuanshou"

    if os.path.exists("datas/submit"):
        shutil.rmtree("datas/submit") # 能删除该文件夹和文件夹下所有文件
    os.mkdir("datas/submit")

    post_process = PostProcess(predict_file, txt_file_dir)
    post_process.conll2brat()


if __name__ == "__main__":
    main()


            
                








