# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """
import logging
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple

import numpy as np
# from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
import torch

logger = logging.getLogger(__name__)


from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask


os.chdir(os.getcwd())


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

LABEL_LIST = [
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
LABEL_LIST.sort()

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters:")
    for key, value in vars(training_args).items():
        logger.info("  %s = %s", key, value)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task
    labels = token_classification_task.get_labels(data_args.labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
        gradient_checkpointing=training_args.gradient_checkpointing
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast,
        additional_special_tokens=["[unused1]"]
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        training_args=training_args
    )
    

    # Get datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            use_crf=training_args.use_crf
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            use_crf=training_args.use_crf
        )
        if training_args.do_eval or training_args.do_predict_dev
        else None
    )

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray, dateset=None) -> Tuple[List[int], List[int]]:
        
        if training_args.use_crf:
            mask = []
            for feature in dateset.features:
                mask.append(feature.attention_mask)
            if mask:
                if isinstance(mask, list):
                    mask = np.array(mask)
                mask = torch.from_numpy(mask).cuda() == 1
                predictions = torch.from_numpy(predictions).cuda()
                preds = model.crf.decode(predictions, mask)
            else:
                preds = model.crf.decode(predictions)

            out_label_list = []
            preds_list = []

            for pred_one, label_one in zip(preds, label_ids.tolist()):
                out_label = []
                pred = []

                for p, l in zip(pred_one, label_one):
                    out_label.append(label_map[l])
                    pred.append(label_map[p])
                    
                out_label = out_label[1:-1]
                pred = pred[1:-1]
                out_label_list.append(out_label)
                preds_list.append(pred)
        else:
            preds = np.argmax(predictions, axis=2)
            batch_size, seq_len = preds.shape
            out_label_list = [[] for _ in range(batch_size)]
            preds_list = [[] for _ in range(batch_size)]

            for i in range(batch_size):
                for j in range(seq_len):
                    if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                        out_label_list[i].append(label_map[label_ids[i][j]])
                        preds_list[i].append(label_map[preds[i][j]])

        
        return preds_list, out_label_list


    def precision_score(y_true, y_pred, average='micro'):
        true_entities = set(y_true)
        pred_entities = set(y_pred)

        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)

        score = nb_correct / nb_pred if nb_pred > 0 else 0

        return score

    def recall_score(y_true, y_pred, average='micro', suffix=False):
        true_entities = set(y_true)
        pred_entities = set(y_pred)

        nb_correct = len(true_entities & pred_entities)
        nb_true = len(true_entities)

        score = nb_correct / nb_true if nb_true > 0 else 0

        return score

    def f_score(y_true, y_pred, average='micro', suffix=False):
        true_entities = set(y_true)
        pred_entities = set(y_pred)

        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        score = 2 * p * r / (p + r) if p + r > 0 else 0

        return score

    def compute_metrics(p: EvalPrediction, mode, dateset=None) -> Dict:
        preds_list, out_label_list = align_predictions(p.predictions, p.label_ids, dateset)
        words_list, preds_list = post_align_predictions(dateset, preds_list, tokenizer)

        reader_file = os.path.join(data_args.data_dir, f"{mode}.txt")

        if mode == "dev":
            data_dir = "datas/brat/train"
        elif mode == "test":
            data_dir = "datas/brat/chusai_xuanshou"
        else:
            raise ValueError(f"mode is error: {mode}")
        
        with open(reader_file, "r", encoding="utf8") as reader:
            pre_tuple, real_tuple = conver_entity_list_to_tuple(reader, preds_list, words_list, data_dir)
        # print(real_tuple[0], pre_tuple[0])
        result = {
            # "accuracy_score": accuracy_score(out_label_list, preds_list),
            "all_precision": precision_score(real_tuple, pre_tuple),
            "all_recall": recall_score(real_tuple, pre_tuple),
            "all_f_score": f_score(real_tuple, pre_tuple),
        }
        for label in LABEL_LIST:
            sub_pre_tuple = [t for t in pre_tuple[:] if t[-1]==label]
            sub_real_tuple = [t for t in real_tuple[:] if t[-1]==label]

            result.update({
                f"{label}_precision": precision_score(sub_real_tuple, sub_pre_tuple),
                f"{label}_recall": recall_score(sub_real_tuple, sub_pre_tuple),
                f"{label}_f_score": f_score(sub_real_tuple, sub_pre_tuple),
            })
        metrics_report = f"\n{'Tag':20s}\t{'Precision':9s}\t{'Recall':9s}\t{'F-Score':9s}\t\n"
        for label in LABEL_LIST:
            pkey = f"{label}_precision"
            rkey = f"{label}_recall"
            fkey = f"{label}_f_score"
            p, r, f = result[pkey], result[rkey], result[fkey]
            metrics_report += f"{label:20s}\t{p:9.7f}\t{r:9.7f}\t{f:9.7f}\t\n"
        metrics_report += "<BLANKLINE>\n"
        pkey = f"all_precision"
        rkey = f"all_recall"
        fkey = f"all_f_score"
        p, r, f = result[pkey], result[rkey], result[fkey]
        label = "ALL"
        metrics_report += f"{label:20s}\t{p:9.7f}\t{r:9.7f}\t{f:9.7f}\t\n"
        logger.info("--------metricd report---------")
        logger.info(metrics_report)

        return result

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_master():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    # logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            use_crf=training_args.use_crf
        )

        # logger.warning(list(zip(tokens_list[3], test_dataset.features[3].input_ids)))
        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids, test_dataset)

        # 数据对齐
        words_list, preds_list = post_align_predictions(test_dataset, preds_list, tokenizer)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_master():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_master():
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, "test.txt"), "r") as f:
                    token_classification_task.write_predictions_to_file(writer, f, preds_list, words_list)


    # Predict dev
    if training_args.do_predict_dev:

        # logger.warning(list(zip(tokens_list[3], test_dataset.features[3].input_ids)))
        predictions, label_ids, metrics = trainer.predict(eval_dataset,  description="Evaluation")
        preds_list, _ = align_predictions(predictions, label_ids, eval_dataset)

        # 数据对齐
        words_list, preds_list = post_align_predictions(eval_dataset, preds_list, tokenizer)

        output_test_results_file = os.path.join(training_args.output_dir, "dev_results.txt")
        if trainer.is_world_master():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "dev_predictions.txt")
        if trainer.is_world_master():
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_args.data_dir, "dev.txt"), "r") as f:
                    token_classification_task.write_predictions_to_file(writer, f, preds_list, words_list)

    return results

def post_align_predictions(dataset, preds_list, tokenizer):
        # 数据对齐
        tokens_list = []
        for feature in dataset.features:
            tokens = tokenizer.convert_ids_to_tokens(feature.input_ids)
            tokens = tokens[1:-1]
            tokens_list.append(tokens)

        preds_decode = []
        tokens_decode = []
        for tokens, preds in zip(tokens_list, preds_list):
            pred_decode = []
            token_decode = []
            for token, pred in zip(tokens, preds):
                if token.startswith("##"):
                    token = token[2:]
                elif token == "[unused1]":
                    token = ["[unused1]"]
                elif token == "[UNK]":
                    token = ["[UNK]"]
                if not isinstance(token, list):
                    token = list(token)
                if pred.startswith("B-"):
                    pred = pred[2:]
                    pred = [f"B-{pred}"] + [f"I-{pred}"]*(len(token)-1)
                else:
                    pred = [pred]*len(token)
                pred_decode.extend(pred)
                token_decode.extend(token)
            preds_decode.append(pred_decode)
            tokens_decode.append(token_decode)
        preds_list = preds_decode
        words_list = tokens_decode

        return words_list, preds_list

import re
from seqeval.metrics.sequence_labeling import get_entities

def conver_entity_list_to_tuple(reader, preds_list, words_list, data_dir):
    '''将预测结果转换为题目要求的元组形式'''
    example_id = 0
    predict_datas = []
    predict_data = []

    real_datas = []
    real_data = []
    for line in reader:
        if line.startswith("-DOCSTART-") or line.strip() == "" or line == "\n":
            # if not preds_list[example_id]:
            example_id += 1
        elif preds_list[example_id]:
            # output_line = line.split()[0] + " " + preds_list[example_id].pop(0) + "\n"
            new_token = words_list[example_id][0]
            new_label = preds_list[example_id][0]
            splits = line.strip().split(" ")
            if len(splits) == 3: # for test data
                orign_label = "O"
                original_token, file_id, pos = splits
            elif len(splits) == 4:
                original_token, file_id, pos, orign_label = splits
            elif len(splits) == 5:
                original_token, file_id, pos, _, orign_label = splits
            else:
                raise ValueError(f"文件存在非法行: {splits}")

            if line.lower().startswith(new_token) or re.search("^\[[a-zA-Z0-9]+\]$", new_token): # for special token:
                words_list[example_id].pop(0)
                preds_list[example_id].pop(0)
                if predict_data and file_id != predict_data[-1]["file_id"]:
                    predict_datas.append(predict_data)
                    predict_data = []
                    real_datas.append(real_data)
                    real_data = []
                
                pre_entity_tuple = {
                    "label": new_label,
                    "pos": pos,
                    "file_id": file_id
                }
                real_entity_tuple = {
                    "label": orign_label,
                    "pos": pos,
                    "file_id": file_id
                }
                if re.search("^\[[a-zA-Z0-9]+\]$", new_token):
                    logger.debug(f"new token is a special token {new_token}")
                else:
                    # 验证是否对齐
                    assert new_token == original_token.lower(), [original_token, new_token]

                predict_data.append(pre_entity_tuple)
                real_data.append(real_entity_tuple)
            else:
                output_line = line.rstrip('\n') + " " + new_label + " " + new_token + " " + f"{int(orign_label==new_label)}" + "\n"
                logger.info(f"new token is deleted by tokenizer: {output_line}")
        else:
            logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line)
    if predict_data:
        predict_datas.append(predict_data)

    pre_entity_tuples = []
    real_entity_tuples = []
    for predict_data in predict_datas:
        file_id = predict_data[0]["file_id"]
        label_list = [data["label"] for data in predict_data]
        pos_list = [data["pos"] for data in predict_data]

        entities = get_entities(label_list)
        pre_entity_tuple = []
        for tag, start, end in entities:
            real_start = int(pos_list[start])
            real_end = int(pos_list[end]) + 1
            pre_entity_tuple.append((file_id, real_start, real_end, tag))
        pre_entity_tuples.extend(pre_entity_tuple) 
        ann_file = os.path.join(data_dir, f"{file_id}.ann")
        if os.path.exists(ann_file):
            with open(ann_file, "r", encoding="utf8") as r:
                for line in r:
                    line = line.strip()
                    if not line:
                        continue
                    _, entity_content, _ = line.split("\t")
                    tag, start, end = entity_content.split(" ")
                    real_entity_tuples.append((file_id, int(start), int(end), tag))
    
    # real_entity_tuples = []
    # for real_data in real_datas:
    #     file_id = real_data[0]["file_id"]
    #     label_list = [data["label"] for data in real_data]
    #     pos_list = [data["pos"] for data in real_data]

    #     entities = get_entities(label_list)
    #     real_entity_tuple = []
    #     for tag, start, end in entities:
    #         real_start = int(pos_list[start])
    #         real_end = int(pos_list[end]) + 1
    #         real_entity_tuple.append((file_id, real_start, real_end, tag))
    #     real_entity_tuples.extend(real_entity_tuple)

    return pre_entity_tuples, real_entity_tuples



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
