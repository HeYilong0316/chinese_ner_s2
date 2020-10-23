import sys
sys.path.append("/home/heyilong/codes/chinese_medical_ner")
print(sys.path)

from ruler.flashtext import KeywordProcessor
from tqdm import tqdm
import pandas as pd

import os
from glob import glob
orignal_brat_dir = "/home/heyilong/codes/chinese_medical_ner/datas/original_brat/train"
brat_dir = "/home/heyilong/codes/chinese_medical_ner/datas/brat/train"




all_equal_freq_label = {"下尿路感染": "DISEASE_GROUP",
                        "中风": "DISEASE",
                        "乳腺囊性增生病": "DISEASE_GROUP",
                        "乳腺纤维瘤": "DISEASE_GROUP",
                        "人肠滴虫": "DISEASE",
                        "体虚": "SYNDROME",
                        "体虚羸弱": "SYNDROME",
                        "健脾燥湿": "DRUG_EFFICACY",
                        "内热瞀闷": "SYNDROME",
                        "再生障碍性贫血": "DISEASE",
                        "出血症": "DISEASE_GROUP",
                        "前列腺增生": "DISEASE",
                        "口干苦": "SYMPTOM",
                        "哮喘": "DISEASE",
                        "四物汤": "DRUG",
                        "外感": "SYNDROME",
                        "妇科病": "DISEASE_GROUP",
                        "实热内盛": "SYNDROME",
                        "小儿盗汗": "SYMPTOM",
                        "局部肿痛": "SYMPTOM",
                        "急性肝损伤": "DISEASE",
                        "慢性肾炎": "DISEASE_GROUP",
                        "慢性附件炎": "DISEASE",
                        "扁桃体炎": "DISEASE",
                        "更年期综合症": "DISEASE_GROUP",
                        "止咳定喘": "DRUG_EFFICACY",
                        "气血俱亏": "SYNDROME",
                        "泌尿结石": "DISEASE_GROUP",
                        "浓茶": "FOOD",
                        "淤血阻滞": "SYNDROME",
                        "湿寒带下": "SYNDROME",
                        "湿热挟瘀症": "DISEASE_GROUP",
                        "湿热淤阻": "SYNDROME",
                        "溃疡性结肠炎": "DISEASE",
                        "溃疡病": "DISEASE_GROUP",
                        "热淋涩痛": "SYMPTOM",
                        "痰湿凝滞": "SYNDROME",
                        "瘀血内阻": "SYNDROME",
                        "瘀阻": "SYNDROME",
                        "瘰疬": "SYNDROME",
                        "精冷而稀": "SYMPTOM",
                        "肝炎": "DISEASE_GROUP",
                        "肝癌": "DISEASE_GROUP",
                        "肺燥咳嗽": "SYMPTOM",
                        "肾亏": "SYNDROME",
                        "肾寒": "SYNDROME",
                        "肾热": "SYNDROME",
                        "肾阳亏损": "SYNDROME",
                        "肿瘤": "DISEASE",
                        "胃癌": "DISEASE_GROUP",
                        "胆囊癌": "DISEASE_GROUP",
                        "脂肪瘤": "DISEASE",
                        "脓肿": "SYMPTOM",
                        "脾胃虚弱": "SYNDROME",
                        "膀胱气化不行": "SYNDROME",
                        "血分有热": "SYNDROME",
                        "血虚萎黄": "SYNDROME",
                        "闪腰岔气": "SYMPTOM",
                        "防止炎症复发": "DRUG_EFFICACY",
                        "阴蚀": "DISEASE",
                        "降低肠管紧张性": "DRUG_EFFICACY",
                        "风寒骨痛": "SYMPTOM",
                        "食道癌": "DISEASE_GROUP",
                        "马兜铃酸": "DRUG_INGREDIENT",
                        "黄芩": "DRUG_INGREDIENT"}


def is_add_dict_entity(dict_ent, ents, text, file_id):
    for ent in ents:
        ent1, ent2 = dict_ent, ent
        ent1_index = set(range(ent1[1], ent1[2]+1))
        ent2_index = set(range(ent2[1], ent2[2]+1))

        if ent1_index == ent2_index:
            continue

        overlap = ent1_index & ent2_index
        if overlap:
            if ent1_index == overlap:
                # 重合，ent2大
                # print(ent1, text[ent1[1]:ent1[2]+1], ent2, text[ent2[1]:ent2[2]+1], file_id, "is_add_dict_entity")
                return False
            elif ent2_index == overlap:
                # 重合 ent1大
                continue
            else:
                # 交集
                # print(ent1, text[ent1[1]:ent1[2]+1], ent2, text[ent2[1]:ent2[2]+1], file_id, "is_add_dict_entity")
                return False
    return True


def is_add_ann_entity(dict_ents, ent, text, file_id):
    for dict_ent in dict_ents:
        ent1, ent2 = dict_ent, ent
        ent1_index = set(range(ent1[1], ent1[2]+1))
        ent2_index = set(range(ent2[1], ent2[2]+1))

        if ent1_index == ent2_index:
            continue

        overlap = ent1_index & ent2_index
        if overlap:
            if ent2_index == overlap:
                # 重合 ent1大
                # print(ent1, text[ent1[1]:ent1[2]+1], ent2, text[ent2[1]:ent2[2]+1], file_id, "is_add_ann_entity")
                return False
    return True


ann_entities = []
all_ann_df = pd.DataFrame()
for txt_file in tqdm(glob(os.path.join(orignal_brat_dir, "*.txt"))):
    file_id = os.path.basename(txt_file).split(".")[0]
    ann_file = os.path.join(orignal_brat_dir, f"{file_id}.ann")
    with open(txt_file, "r", encoding="utf8") as r:
        text = r.read()
    with open(ann_file, "r", encoding="utf8") as r:
        for line in r:
            _id, label, start, end, mention = line.strip().split()
            start = int(start)
            end = int(end)
            mention = text[start:end]
            ann_entities.append([mention, label, [start, end-1]])
all_ann_df = pd.DataFrame(ann_entities)
all_ann_df.columns = ["mention", "label", "span"]

ann_mention_gp = all_ann_df.drop("span", axis=1).groupby(
    "mention").agg(lambda x: list(x)).reset_index()


def get_max_freq_label(row):
    labels = row.label
    mention = row.mention
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    ret = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    if len(ret) > 1 and ret[0][1] == ret[1][1]:
        print('"{}": {},'.format(mention, " ".join(
            ['"{}"'.format(r[0]) for r in ret[:2]])))
        # print(ret, mention)
    return ret[0][0]


ann_mention_gp["label"] = ann_mention_gp.apply(get_max_freq_label, axis=1)
all_mention2label_map = ann_mention_gp.set_index(
    "mention").T.to_dict(orient="records")[0]
all_mention2label_map.update(all_equal_freq_label)


for txt_file in glob(os.path.join(orignal_brat_dir, "*.txt")):
    file_id = os.path.basename(txt_file).split(".")[0]
    ann_file = os.path.join(orignal_brat_dir, f"{file_id}.ann")
    with open(txt_file, "r", encoding="utf8") as r:
        text = r.read()
    extractor = KeywordProcessor()
    entity_dict = {}
    ann_entities = []
    mention_label_count = {}

    with open(ann_file, "r", encoding="utf8") as r:
        for line in r:
            _id, label, start, end, mention = line.strip().split()
            start = int(start)
            end = int(end)
            mention = text[start:end]
            ann_entities.append([mention, label, [start, end-1]])
    ann_df = pd.DataFrame(ann_entities)
    ann_df.columns = ["mention", "label", "span"]

    # 统计每个mention对应的label频率
    ann_mention_gp = ann_df.drop("span", axis=1).groupby("mention").agg(lambda x: list(x)).reset_index()

    def get_max_freq_label(row):
        labels = row.label
        mention = row.mention
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        ret = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        if len(ret) > 1 and ret[0][1] == ret[1][1]:
            # print("{} {}".format(mention, " ".join([r[0] for r in ret])))
            return None
        return ret[0][0]
    ann_mention_gp["label"] = ann_mention_gp.apply(get_max_freq_label, axis=1)
    mention2label_map = ann_mention_gp.set_index("mention").T.to_dict(orient="records")[0]

    # mention选取频率最高的label
    ann_df["label"] = ann_df.mention.apply(lambda x: mention2label_map[x] if mention2label_map[x] is not None else all_mention2label_map[x])
    # ann_df["label"] = ann_df.mention.apply(lambda x: mention2label_map[x] if mention2label_map[x] is not None else mention2label_map[x])

    label2mentions_map = ann_df.drop("span", axis=1).groupby("label").agg(
        lambda x: list(set(x))).T.to_dict(orient="records")[0]
    extractor.add_keywords_from_dict(label2mentions_map)

    dict_entities = extractor.extract_keywords(text, span_info=True)
    ann_entities = [(a[1], a[2][0], a[2][1])
                    for a in ann_df.to_records(index=None)]

    entities = []
    for dict_ent in dict_entities:
        if is_add_dict_entity(dict_ent, ann_entities, text, file_id):
            entities.append(dict_ent)

    for ann_ent in ann_entities:
        if is_add_ann_entity(dict_entities, ann_ent, text, file_id):
            entities.append(ann_ent)

    entities = sorted(list(set(entities)), key=lambda x: (x[1], x[2]))

    to_save = ""
    for i, (label, real_start, real_end) in enumerate(entities):
        real_end += 1
        to_save += f"T{i}\t{label} {real_start} {real_end}\t{text[real_start:real_end]}\n"
    to_save_file = os.path.join(brat_dir, f"{file_id}.ann")
    with open(to_save_file, "w", encoding="utf8") as w:
        w.write(to_save)
