# html删掉
# 逗号 引号 冒号 分号 括号中文转英文
# 尚不明确删掉
# (), [], <>里面全英文的


from seqeval.metrics.sequence_labeling import get_entities
import logging
import pandas as pd
import os
import re
import sys
from glob import glob
import json


logger = logging.getLogger(__name__)


MAX_LEN = 510

LABELS_LIST = [
    "DRUG",
    "DRUG_INGREDIENT",
    "DISEASE",
    "SYMPTOM",
    "SYNDROME",
    "DISEASE_GROUP",
    "FOOD",
    "FOOD_GROUP",
    "PERSON_GROUP",
    "DRUG_GROUP",
    "DRUG_DOSAGE",
    "DRUG_TASTE",
    "DRUG_EFFICACY"
]


BLANK_RE = re.compile("\s")

ZH_RE = re.compile("[\u4e00-\u9fa5]")

REPLACE_MAP = {
    "×": "*",
    "X": "*",
    "“": '"',
    "”": '"',
    "〈": "<",
    "《": "《",
    "》": "》",
    "）": ")",
    "，": ",",
    "．": ".",
    "：": ":",
    "；": ";",
    '＜': "<",
    '［': "[",
    '］': "]",
    '～': "~",
    "（": "(",
    "-": "~",
    '１': "1",
    '２': "2",
    '３': "3",
    '７': "7",
    '／': "/",
    "+": " ",
    "·": " ",
    '―': "~",
    '≤': "<",
    '≥': ">",
    '！': "。",
    '％': "%",
    '－': "~"
}

REMOVE_LIST = [
    # '"',
    # '#',
    # '%',
    "'",
    "】",
    '*',
    '/',
    '—',
    '…',
    '℃',
    '∶',
    '⑸',
    '⑺',
    '⑼',
    '⑽',
    '⑾',
    '⑿',
    '⒀',
    '⒈',
    '⒉',
    '⒊',
    '⒋',
    '⒌',
    '⒍',
    '⒎',
    '⒏',
    '⒐',
    '⒑',
    '⒒',
    '⒖',
    '┦',
    '\u3000',
    'ぁ',
    'は',
    'ク',
    'ダ',
    'ヌ',
    'ヒ',
    'プ',
    '\ue000',
    '\ue002',
    '\ue004',
    '\ue01a',
    '\ue027',
    '\ue02e',
    '\ue034',
    '\ue04f',
    '\ue060',
    '\ue074',
    '\ue093',
    '\ue0bc',
    '\ue0be',
    '\ue0ce',
    '\ue0d2',
    '\ue0d4',
    '\ue0d6',
    '\ue0d7',
    '\ue0de',
    '\ue0df',
    '\ue0e1',
    '\ue0e9',
    '\ue0ea',
    '\ue0ec',
    '\ue0ed',
    '\ue0ee',
    '\ue132',
    '\ue13f',
    '\ue1d8',
    '\ue1e6',
    '\ue1fb',
    '\ue207',
    '\ue225',
    '\ue246',
    '\ue262',
    '\ue264',
    '\ue269',
    '\ue292',
    '\ue2be',
    '\ue2c7',
    '\ue312',
    '\ue380',
    '\ue383',
    '\ue3da',
    '\ue422',
    '\ue42c',
    '\ue431',
    '\ue432',
    '\ue456',
    '\ue468',
    '\ue46a',
    '\ue49a',
    '\ue5cb',
    '\ue787',
    '\ue800',
    '＃',
]

REFINE_RE = [
    r"&[a-zA-Z0-9]+;",
    r"(\s+)(?=\s)",
    r"(\.+)(?=\.)",
    r"([0.]+)(?=0)",
    r"\d+[.、)](?=\D)",
    r"\?.*?\?",
]


REMOVE_RE = [

]

print(REMOVE_RE)

# 获取一次多意的实体


def get_polysemy():
    train_dir_path = "../user_data/train"
    train_file_list = glob(os.path.join(train_dir_path, "*.txt"))
    polysemy_dict = {}
    for file_name in train_file_list:
        with open(file_name, "r", encoding="utf8") as r:
            text = r.read()
        file_name = file_name[:-3] + "ann"
        with open(file_name, "r", encoding="utf8") as r:
            for line in r:
                line = line.strip()
                if not line:
                    continue
                else:
                    _, content, mention = line.split("\t")
                    tag, start, end = content.split(" ")
                    start = int(start)
                    end = int(end)
                    polysemy_dict[text[start:end]] = polysemy_dict.get(
                        text[start:end], set()) | set([tag])

    polysemy_set = set()
    for mention, tags in polysemy_dict.items():
        if len(tags) > 1:
            polysemy_set.add(mention)
    print(polysemy_set)
    return polysemy_set


# polysemy_set = get_polysemy()
polysemy_set = set()


class PreProcessBase:
    def __init__(self, txt_file, ann_file=None):
        # read file
        file_id = os.path.basename(txt_file).split(".")[0]
        with open(txt_file, "r", encoding="utf8") as r:
            text = r.read()
        if ann_file:
            ann = pd.read_csv(ann_file, sep="\t", header=None)
            ann.columns = ["id_", "label_and_span", "mention"]
            ann["label_and_span"] = ann.label_and_span.apply(
                lambda x: x.strip().split(" "))
            ann["label"] = ann.label_and_span.apply(lambda x: x[0])
            ann["span"] = ann.label_and_span.apply(
                lambda x: list(map(lambda x: int(x), x[1:])))
            ann = ann.drop("label_and_span", axis=1)
            ann["mention"] = ann.span.apply(lambda x: text[x[0]:x[1]])
            ann = ann[~ann.mention.isin(polysemy_set)]
        else:
            ann = None

        # 所有空格转英文空格
        text = BLANK_RE.sub(" ", text)
        # 替换特殊字符
        for k, v in REPLACE_MAP.items():
            text = text.replace(k, v)

        text = text.lower()
        text = re.sub(r"(\d)[,、? ](\D)", r"\1.\2", text)
        text = re.sub(r"\d", r"0", text)
        text = re.sub(r"([\u4e00-\u9fa5])\. ", r"\1。 ", text)
        text = re.sub(r"([\u4e00-\u9fa5])\.([\u4e00-\u9fa5])", r"\1,\2", text)
        text = text.replace("/td>", "<td>")
        text = text.replace("<br/>", "     ")

        self.text = text
        self.ann = ann
        self.file_id = file_id

    def _brat2conll(self, text, ann):
        conll_text = []
        conll_label = []
        if ann is not None:
            for _, row in ann.iterrows():
                label = row.label
                assert label in LABELS_LIST, [label]
                start, end = row.span

                text = list(text)
                text[start] = f"<{label}>" + text[start]
                text[end-1] = text[end-1] + f"</{label}>"

        is_inner = False
        cur_label = None
        # brat转为conll格式
        for word in text:
            match_start = re.search(f"^<({'|'.join(LABELS_LIST)})>", word)
            match_end = re.search(f"</({'|'.join(LABELS_LIST)})>$", word)
            if match_start:
                label = match_start.group(1)
                assert label in LABELS_LIST, [label]
                conll_label.append(f"B-{label}")
                if not match_end:
                    is_inner = True
                cur_label = label
            elif is_inner:
                conll_label.append(f"I-{cur_label}")
                if match_end:
                    is_inner = False
                    cur_label = None
            else:
                conll_label.append("O")

            word = re.sub(f"</?({'|'.join(LABELS_LIST)})>", "", word)
            conll_text.append(word)
        return conll_text, conll_label

    def convert_char_to_special_token(self, sentences, char, token):
        '''将token转为特殊token'''
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                if sentences[i][j][0] == char:
                    sentences[i][j][0] = token
        return sentences

    def segment_sentences(self, conll_text, poses, conll_label, file_id, ann=True):
        '''分句'''
        sentences = []
        sentence = []
        for i, (word, pos, label) in enumerate(zip(conll_text, poses, conll_label)):
            if (
                    i >= 2
                and
                    word != " "
                and
                    conll_text[i-1] == " "
                and
                    conll_text[i-2] == " "
            ):
                # 如果连续两个以上空格就分句
                sentences.append(sentence)
                if ann is not None:
                    sentence = [[word, file_id, pos, label]]
                else:
                    sentence = [[word, file_id, pos]]
            elif (
                    word.isdigit()
                and
                    (i == 0 or not conll_text[i-1].isdigit())
                and
                    i < (len(conll_text)-3)
                and
                    conll_text[i+1].isdigit()
                and
                    conll_text[i+2] in ".,)"
                and not
                    conll_text[i+3].isdigit()
            ):
                # 11. 12. 13. ...这种序号的需要分句
                sentences.append(sentence)
                if ann is not None:
                    sentence = [[word, file_id, pos, label]]
                else:
                    sentence = [[word, file_id, pos]]
            elif (
                    word.isdigit()
                and
                    (i == 0 or not conll_text[i-1].isdigit())
                and
                    i < (len(conll_text)-2)
                and
                    conll_text[i+1] in ".,)"
                and not
                    conll_text[i+2].isdigit()
            ):
                # 1. 2. 3. ...这种序号的需要分句
                sentences.append(sentence)
                if ann is not None:
                    sentence = [[word, file_id, pos, label]]
                else:
                    sentence = [[word, file_id, pos]]
            elif word in "。;":
                # 标点分句
                if ann is not None:
                    sentence += [[word, file_id, pos, label]]
                else:
                    sentence += [[word, file_id, pos]]
                sentences.append(sentence)
                sentence = []
            else:
                if ann is not None:
                    sentence += [[word, file_id, pos, label]]
                else:
                    sentence += [[word, file_id, pos]]
        if sentence:
            sentences.append(sentence)
        if MAX_LEN < 0:
            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)
        return sentences

    @staticmethod
    def strip(string_list, regexp):
        rets = []
        for string in string_list:
            while string and re.search(regexp, string[0][0]):
                string.pop(0)
            while string and re.search(regexp, string[-1][0]):
                string.pop()
            rets.append(string)
        return rets

    def filter_zh_sentences(self, sentences):
        '''没出现中文的句子和只有一个字的句子删掉'''
        ret = []
        # 一个中文都没有的句子删掉
        for sentence in sentences:
            if len(sentence) > 1 and ZH_RE.search("".join(c[0] for c in sentence)):
                # if ZH_RE.search("".join(c[0] for c in sentence)):
                ret.append(sentence)
        return ret

    def assert_pos(self, conll):
        '''检查和brat格式数据位置是否对齐'''
        text = self.text
        for terms in conll:
            for term in terms:
                word, pos = term[0], term[2]
                if word != "[unused1]":
                    assert word == text[pos], [term, text[pos]]

    def get_remove_index(self, regexps, string, whole=False):
        '''根据正则删除句子里的一些token'''
        remove_index_list = set()
        string_copy = string[:]
        for regexp in regexps:
            finditer = re.finditer(regexp, string)
            for search in finditer:
                start, end = search.span()
                remove_index_list.update(list(range(start, end)))
            string_copy = re.sub(regexp, "", string_copy)

        remove_index_list = list(remove_index_list)
        remove_index_list.sort()
        return remove_index_list

    def combine_sentences(self, sentences):
        rets = []
        ret = []
        for sentence in sentences:
            if len(ret) + len(sentence) <= MAX_LEN:
                ret += sentence
            else:
                rets.append(ret)
                ret = sentence
        if ret:
            rets.append(ret)
        return rets


class PreProcess(PreProcessBase):
    def brat2conll(self):
        '''brat格式转conll'''
        text = self.text
        ann = self.ann
        file_id = self.file_id
        info = {}

        conll_text = []
        conll_label = []
        info = {}

        # brat转conll格式
        conll_text, conll_label = self._brat2conll(text, ann)

        assert len(conll_text) == len(conll_label)

        # 预处理
        sentences = list(zip(conll_text, list(
            range(len(conll_text))), conll_label))
        sentences = self.pre_segment_preprocess(sentences)
        if not sentences:
            return sentences, {"del_sen": 0}
        conll_text, poses, conll_label = zip(*sentences)

        # 分句
        sentences = self.segment_sentences(
            conll_text, poses, conll_label, file_id, ann is not None)
        if not sentences:
            return sentences, {"del_sen": 0}
        length_segment = len(sentences)

        # 后处理
        sentences = self.post_segment_preprocess(sentences)
        if not sentences:
            return sentences, {"del_sen": 0}
        length_post = len(sentences)
        info["del_sen"] = length_segment - length_post

        if MAX_LEN > 0:
            sentences = self.combine_sentences(sentences)

            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)

        sentences = self.convert_char_to_special_token(
            sentences, " ", "[unused1]")

        # 检查位置是否对应
        self.assert_pos(sentences)

        return sentences, info

    def pre_segment_preprocess(self, sentence):
        '''分句前的处理'''

        string = "".join([c[0] for c in sentence])
        # 删除括号里是英文的和html等
        remove_index_list = self.get_remove_index(
            ["[<\[(][^\u4e00-\u9fa5]*?[>\])]"], string)
        # 删除无关字符
        for index, c in enumerate(string):
            if c in REMOVE_LIST:
                remove_index_list.append(index)

        remove_index_list = list(remove_index_list)
        remove_index_list.sort()

        sentence = self.remove_from_index(sentence, string, remove_index_list)

        return sentence

    def remove_from_index(self, sentence, string, remove_index_list):
        tmp = []
        bad_index_start = None
        for index, term in enumerate(sentence):
            if index not in remove_index_list:
                if bad_index_start is not None:
                    print(string)
                    print((bad_index_start, index))
                    print(string[bad_index_start:index])
                    print(" ")
                    print([s[-1] for s in sentence[bad_index_start:index]])
                    bad_index_start = None
                tmp.append(term)
            else:
                if bad_index_start is not None:
                    continue
                elif sentence[index][-1] != "O":
                    bad_index_start = index
        if bad_index_start is not None:
            print(string)
            print((bad_index_start, len(sentence)))
            print(string[bad_index_start:])
            print([s[-1] for s in sentence[bad_index_start:]])
            print(" ")
        sentence = tmp
        return sentence

    def post_segment_preprocess(self, sentences):
        '''分句后的处理'''

        # refine文本
        for i, sentence in enumerate(sentences):
            string = "".join([c[0] for c in sentence])
            remove_index_list = self.get_remove_index(
                REFINE_RE, string, whole=True)
            for index, c in enumerate(string):
                if c in REMOVE_LIST:
                    remove_index_list.append(index)

            remove_index_list = list(remove_index_list)
            remove_index_list.sort()

            sentence = self.remove_from_index(
                sentence, string, remove_index_list)
            sentences[i] = sentence

        # 删除一些无关句子
        for i, sentence in enumerate(sentences):
            string = "".join([c[0] for c in sentence])
            remove_index_list = self.get_remove_index(
                REMOVE_RE, string, whole=True)

            remove_index_list = list(remove_index_list)
            remove_index_list.sort()

            sentence = self.remove_from_index(
                sentence, string, remove_index_list)
            sentences[i] = sentence

        if MAX_LEN < 0:
            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)
        return sentences


def run(file_name_list, mode):
    import datetime
    print(f"------------------start For {mode}----------------------")
    conlls = []
    all_del_sen = 0
    all_label_num = 0
    for file_name in sorted(file_name_list):
        txt_name = file_name
        if mode != "test":
            ann_name = file_name[:-4] + ".ann"
            preprocess = PreProcess(txt_name, ann_name)
            all_label_num += preprocess.ann.shape[0]
        else:
            preprocess = PreProcess(txt_name)
        conll, info = preprocess.brat2conll()
        all_del_sen += info["del_sen"]
        conlls.extend(conll)

    max_len = max([len(s) for s in conlls])
    label_num = 0
    for conll in conlls:
        label = [c[-1] for c in conll]
        label_num += len(get_entities(label))
    print(
        f'''
            句子总数   : {len(conlls)}
            删除句子个数: {all_del_sen}
            句子最大长度: {max_len}
            原始实体个数: {all_label_num}
            当前实体个数: {label_num}
        '''
    )
    # 打印lanbel的分布情况
    label_dict = {}
    for sentence in conlls:
        label = [s[-1] for s in sentence]
        for entity in get_entities(label):
            entity = entity[0]
            label_dict[entity] = label_dict.get(entity, 0) + 1
    if mode != "test":
        print("实体分布情况：")
        totle_num = 0
        for l in LABELS_LIST:
            if l not in label_dict:
                label_dict.update({l: 0})
        totle_num = sum(label_dict.values())
        for k, v in sorted(label_dict.items(), key=lambda x: x[0]):
            print(f"{k}:\t{v}\t{v/totle_num}")
        print(f"All: {totle_num}")

    print("-------------------END----------------------\n")
    return conlls


def save_to_file(conlls, path, mode):
    if not os.path.exists(path):
        os.mkdir(path)

    string = ""
    for conll in conlls:
        for term in conll:
            term = [str(t) for t in term]
            string += " ".join(term) + "\n"
        string += "\n"

    output_file = os.path.join(path, f"{mode}.txt")
    with open(output_file, "w", encoding="utf8") as w:
        w.write(string)
    return conlls


def gen_count_for_data(conlls_train, conlls_dev, conlls_test):
    # 生成句子分布
    text_list = []
    for conll in conlls_train + conlls_dev + conlls_test:
        text_list.append("".join(c[0] for c in conll))

    text_df = pd.DataFrame(text_list)
    text_df.columns = ["text"]
    text_df.text.value_counts().reset_index().to_csv(
        "../user_data/conll/text_counts_ori.csv", index=None)

    # 生成句子标注的分布
    text_label_dict = {}
    text_count_dict = {}
    for conll in conlls_train + conlls_dev:
        text = "".join(c[0] for c in conll)
        # 记录句子的频数
        text_count_dict[text] = text_count_dict.get(text, 0) + 1
        # 记录每个句子的实体的频数
        entity = get_entities([c[-1] for c in conll])
        entity = [(tag, start, end, text[start:end+1])
                  for tag, start, end in entity]
        value = text_label_dict.get(text, {})
        for e in entity:
            value[e] = value.get(e, 0) + 1
        text_label_dict[text] = value

    def is_conflict(label1, label2):
        if set(range(int(label1[1]), int(label1[2])+1)) & set(range(int(label2[1]), int(label2[2])+1)):
            return True
        else:
            return False

    # 纠正每个句子的标注
    text_entities_map = {}
    for text, entity_dict in text_label_dict.items():
        # 位置考前频数大的排前面
        entities = []
        entity_list = sorted(entity_dict.items(), key=lambda x: (x[0][1]))

        pre_freq = None
        for entity, freq in entity_list:
            # if freq / text_count_dict[text] < 0.1:
            #     # 小于一定频次的标注直接删掉
            #     continue
            if entities and is_conflict(entities[-1], entity):
                # 有冲突: 如果频数相同，取长的那个
                if (freq == pre_freq and (entity[2] - entity[1]) > (entities[-1][2] - entities[-1][1])) or (freq > pre_freq):
                    entities[-1] = entity
                    pre_freq = freq
            else:
                pre_freq = freq
                entities.append(entity)
        text_entities_map[text] = entities

    text_entities_list = [{"text": text, "label": [
        list(e) for e in label]} for text, label in text_entities_map.items()]
    text_entities_list = json.dumps(
        text_entities_list, ensure_ascii=False, indent=2)
    with open("../ser_data/conll/text_label.json", "w", encoding="utf8") as w:
        w.write(text_entities_list)

    return text_entities_map


def repair_label(conll, text_entities_map):
    words, file_id, pos, ori_label = zip(*conll)
    text = "".join(words)
    entity = text_entities_map.get(text, [])
    label = ["O" for _ in range(len(words))]
    for e in entity:
        tag, start, end, _ = e
        start = int(start)
        end = int(end)
        label[e[1]] = f"B-{tag}"
        for i in range(start+1, end+1):
            label[i] = f"I-{tag}"
    conll = zip(words, file_id, pos, ori_label, label)
    return conll


def main_single():
    from sklearn.model_selection import train_test_split

    train_dir_path = "data/train"
    train_file_list = glob(os.path.join(train_dir_path, "*.txt"))

    output_dir = os.path.join("../user_data", "conll")
    if os.path.exists(output_dir):
        raise FileExistsError(f"file exit: {output_dir}")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 2333
    # 4018
    # 划分训练集和验证集
    train_file_list, dev_file_list = train_test_split(
        train_file_list, test_size=0.2, random_state=2333)

    conlls_train = run(train_file_list, "train")
    conlls_dev = run(dev_file_list, "dev")
    # 测试机
    test_dir_path = "/tcdata/juesai"
    test_file_list = glob(os.path.join(test_dir_path, "*.txt"))
    conlls_test = run(test_file_list, "test")

    save_to_file(conlls_train, output_dir, "train")
    save_to_file(conlls_dev, output_dir, "dev")
    save_to_file(conlls_test, output_dir, "test")


def main_k_fold(mode):
    from sklearn.model_selection import KFold
    import numpy as np

    k_fold = KFold(n_splits=10, shuffle=True, random_state=2333)

    test_dir_path = "/tcdata/juesai"
    # test_dir_path = "/home/heyilong/codes/chinese_medical_ner/user_data/chusai_xuanshou"
    test_file_list = glob(os.path.join(test_dir_path, "*.txt"))

    conlls_test = run(test_file_list, "test")

    k_fold_dir = os.path.join("../user_data", "k_fold")
    if not os.path.exists(k_fold_dir):
        os.mkdir(k_fold_dir)
    save_to_file(conlls_test, k_fold_dir, "test")

    train_dir_path = "../user_data/train"
    train_file_list = glob(os.path.join(train_dir_path, "*.txt"))
    train_file_array = np.array(train_file_list)

    if mode != "predict":    
        for i, (train_list, dev_list) in enumerate(k_fold.split(train_file_list)):
            print(f"for {i}-fold")
            output_dir = os.path.join(k_fold_dir, f"fold_{i}")   
            train_list = train_file_array[train_list].tolist()
            dev_list = train_file_array[dev_list].tolist()
            conlls_train = run(train_list, "dev")
            conlls_dev = run(dev_list, "dev")

            save_to_file(conlls_train, output_dir, "train")
            save_to_file(conlls_dev, output_dir, "dev")
        


if __name__ == "__main__":
    if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1]) == "single":
        print("run single")
        main_single()

    elif len(sys.argv) > 1 and sys.argv[1] == "k-fold":
        print("run k-fold")
        main_k_fold(sys.argv[2])
    else:
        raise ValueError("参数错误")
