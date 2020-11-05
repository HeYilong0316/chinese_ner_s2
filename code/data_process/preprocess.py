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

sys.path = [os.getcwd()] + sys.path
logger = logging.getLogger(__name__)


user_data_dir = "../user_data/data"
test_dir_path = "/tcdata/juesai"
# test_dir_path = "/home/heyilong/codes/chinese_medical_ner/user_data/chusai_xuanshou"
train_dir_path = "../user_data/data/train"


try:
    K_FOLD = int(sys.argv[1])
    MAX_LEN = int(sys.argv[2])
except Exception as e:
    print(e)
    K_FOLD = 10
    MAX_LEN = 512

print(f"K_FOLD: {K_FOLD}, MAX_LEN: {MAX_LEN}")

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

SPECIAL_RE = re.compile("[^\u4e00-\u9fa5_a-z_A-Z_0-9]")

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

KEEP_LIST = [' ', '"', '#', '%', '&', '(', ')', ',', '.', ':', ';', '<', '=', '>', '?', '~', '、', '。']

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
            if ZH_RE.search("".join(c[0] for c in sentence)):
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
            if SPECIAL_RE.match(c) and (c not in KEEP_LIST):
                remove_index_list.append(index)

        remove_index_list = list(remove_index_list)
        remove_index_list.sort()

        if remove_index_list:
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
                if SPECIAL_RE.match(c) and (c not in KEEP_LIST):
                    remove_index_list.append(index)

            remove_index_list = list(remove_index_list)
            remove_index_list.sort()

            if remove_index_list:
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

            if remove_index_list:
                sentence = self.remove_from_index(
                    sentence, string, remove_index_list)
            sentences[i] = sentence

        if MAX_LEN < 0:
            sentences = self.strip(sentences, " ")
            sentences = self.filter_zh_sentences(sentences)
        return sentences


def run(file_name_list, lexicon_extractor, mode):
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

    max_len = max([len(s)+2 for s in conlls])
    
    # 加入lexicon之后的最大长度
    max_lexicon_len = 0
    for conll in conlls:
        string = "".join([s[0] for s in conll]).replace("[unused1]", " ")
        length = len(conll) + 2
        lexicons = lexicon_extractor.extract(string)
        for lexicon in lexicons:
            lexicon = lexicon[0]
            length += len(lexicon) + 1
        if  max_lexicon_len < length:
            max_lexicon_len = length

    label_num = 0
    for conll in conlls:
        label = [c[-1] for c in conll]
        label_num += len(get_entities(label))

    print(
        f'''
            句子总数   : {len(conlls)}
            删除句子个数: {all_del_sen}
            句子最大长度: {max_len}
            加入lexicon最大长度：{max_lexicon_len}
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
    print(f"save to {output_file}")
    return conlls


def make_lexicon():
    lexicon = set()
    # 储存字典
    dict_path = os.path.join(user_data_dir, "dicts")
    if not os.path.exists(dict_path):
        os.mkdir(dict_path)
    for txt_file in glob(os.path.join(train_dir_path, "*.txt")):
        with open(txt_file, "r", encoding="utf8") as r:
            text = r.read()
        ann_file = os.path.join(train_dir_path, os.path.basename(txt_file).split(".")[0] + ".ann")
        ann = pd.read_csv(ann_file, sep="\t", header=None)
        ann.columns = ["id_", "label_and_span", "mention"]
        ann["label_and_span"] = ann.label_and_span.apply(
            lambda x: x.strip().split(" "))
        ann["label"] = ann.label_and_span.apply(lambda x: x[0])
        ann["span"] = ann.label_and_span.apply(
            lambda x: list(map(lambda x: int(x), x[1:])))
        ann = ann.drop("label_and_span", axis=1)
        ann["mention"] = ann.span.apply(lambda x: text[x[0]:x[1]])
        lexicon.update(ann.mention.values.tolist())

    lexicon = sorted(list(lexicon))
    with open(os.path.join(dict_path, "lexicon.json"), "w", encoding="utf8") as w:
        json.dump(lexicon, w, ensure_ascii=False, indent=2)
    return lexicon

def main_k_fold(mode):
    from sklearn.model_selection import KFold
    import numpy as np
    from ruler.flashtext import KeywordProcessor


    k_fold_dir = os.path.join(user_data_dir, f"k_fold_{K_FOLD}_{MAX_LEN}")
    if not os.path.exists(k_fold_dir):
        os.mkdir(k_fold_dir)

    from ruler.lexicon_extractor import lexicon_extractor

    k_fold = KFold(n_splits=K_FOLD, shuffle=True, random_state=2333)

    if "predict" in mode:
        test_file_list = glob(os.path.join(test_dir_path, "*.txt"))
        conlls_test = run(test_file_list, lexicon_extractor, "test")
        save_to_file(conlls_test, k_fold_dir, "test")

    train_file_list = glob(os.path.join(train_dir_path, "*.txt"))
    train_file_array = np.array(train_file_list)
    

    if "train" in mode:
        for i, (train_list, dev_list) in enumerate(k_fold.split(train_file_list)):
            print(f"for {i}-fold")
            output_dir = os.path.join(k_fold_dir, f"fold_{i}")   
            train_list = train_file_array[train_list].tolist()
            dev_list = train_file_array[dev_list].tolist()
            conlls_train = run(train_list, lexicon_extractor, "train")
            conlls_dev = run(dev_list, lexicon_extractor, "dev")

            save_to_file(conlls_train, output_dir, "train")
            save_to_file(conlls_dev, output_dir, "dev")    

if __name__ == "__main__":
    mode = sys.argv[3]
    main_k_fold(mode)

