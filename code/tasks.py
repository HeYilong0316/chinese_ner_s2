import logging
import os
from typing import List, TextIO, Union

from conllu import parse_incr

from utils_ner import InputExample, Split, TokenClassificationTask

import re


logger = logging.getLogger(__name__)

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
    "DRUG_EFFICACY",
]

tmp = ["O"]
for label in LABELS_LIST:
    tmp.extend([f"B-{label}", f"I-{label}"])
LABELS_LIST = tmp
logger.info(LABELS_LIST)

class NER(TokenClassificationTask):
    def __init__(self, label_idx=-1):
        # in NER datasets, the last column is usually reserved for NER label
        self.label_idx = label_idx

    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            f = f.read()
            for sentence in f.split("\n\n"):
                if sentence.startswith("-DOCSTART-") or not sentence.strip():
                    continue
                words = []
                labels = []
                last_label = ""

                for line in sentence.split("\n"):
                    splits = line.strip().split(" ")

                    # if mode != "test":
                    label = splits[-1]
                    if len(splits) == 4:
                        token = splits[0]
                    # else:
                    #     label = "O"
                    #     if len(splits) == 3:
                    #         token = splits[0]
                    if label == "O" and last_label == "O":
                        words[-1] += token
                    elif label == "O" and last_label != "O":
                        words.append(token)
                        labels.append("O")
                    elif label.startswith("B-"):
                        words.append(token)
                        # if label[2:] == "DRUG_GROUP":
                        #     labels.append("O")
                        # else:
                        labels.append(label[2:])
                    else:
                        words[-1] += token
                    last_label = label
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                guid_index += 1
        # logger.warning(list(zip(examples[3].words, examples[3].labels)))
        
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List, words_list: List):
        example_id = 0
        for line in test_input_reader:
            if line.startswith("-DOCSTART-") or line.strip() == "" or line == "\n":
                writer.write(line)
                # if not preds_list[example_id]:
                example_id += 1
            elif preds_list[example_id]:
                # output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
                new_token = words_list[example_id][0]
                new_label = preds_list[example_id][0]
                orign_label = line.rstrip('\n').split(" ")[-1]
                if line.lower().startswith(new_token) or re.search("^\[[a-zA-Z0-9]+\]$", new_token):
                    output_line = line.rstrip('\n') + " " + preds_list[example_id].pop(0) + " " + words_list[example_id].pop(0) + " " + f"{int(orign_label==new_label)}" + "\n"
                    writer.write(output_line)
                else:
                    # 被bert的tokenizer删除的字符
                    output_line = line.rstrip('\n') + " " + new_label + " " + new_token + " " + f"{int(orign_label==new_label)}" + "\n"
                    logger.info(f"deleted by tokenizer: {output_line}")
                
            else:
                output_line = line
                writer.write(output_line)
                logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line)

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return LABELS_LIST
            # return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


class Chunk(NER):
    def __init__(self):
        # in CONLL2003 dataset chunk column is second-to-last
        super().__init__(label_idx=-2)

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                labels = f.read().splitlines()
            if "O" not in labels:
                labels = ["O"] + labels
            return labels
        else:
            return [
                "O",
                "B-ADVP",
                "B-INTJ",
                "B-LST",
                "B-PRT",
                "B-NP",
                "B-SBAR",
                "B-VP",
                "B-ADJP",
                "B-CONJP",
                "B-PP",
                "I-ADVP",
                "I-INTJ",
                "I-LST",
                "I-PRT",
                "I-NP",
                "I-SBAR",
                "I-VP",
                "I-ADJP",
                "I-CONJP",
                "I-PP",
            ]


class POS(TokenClassificationTask):
    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        if isinstance(mode, Split):
            mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []

        with open(file_path, encoding="utf-8") as f:
            for sentence in parse_incr(f):
                words = []
                labels = []
                for token in sentence:
                    words.append(token["form"])
                    labels.append(token["upos"])
                assert len(words) == len(labels)
                if words:
                    examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                    guid_index += 1
        return examples

    def write_predictions_to_file(self, writer: TextIO, test_input_reader: TextIO, preds_list: List):
        example_id = 0
        for sentence in parse_incr(test_input_reader):
            s_p = preds_list[example_id]
            out = ""
            for token in sentence:
                out += f'{token["form"]} ({token["upos"]}|{s_p.pop(0)}) '
            out += "\n"
            writer.write(out)
            example_id += 1

    def get_labels(self, path: str) -> List[str]:
        if path:
            with open(path, "r") as f:
                return f.read().splitlines()
        else:
            return [
                "ADJ",
                "ADP",
                "ADV",
                "AUX",
                "CCONJ",
                "DET",
                "INTJ",
                "NOUN",
                "NUM",
                "PART",
                "PRON",
                "PROPN",
                "PUNCT",
                "SCONJ",
                "SYM",
                "VERB",
                "X",
            ]
