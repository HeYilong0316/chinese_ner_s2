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
""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    lexicon_mask: Optional[List[int]] = None
    position_ids: Optional[List[int]] = None
    words_id: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class TokenClassificationTask:
    def read_examples_from_file(self, data_dir, mode: Union[Split, str]) -> List[InputExample]:
        raise NotImplementedError

    def get_labels(self, path: str) -> List[str]:
        raise NotImplementedError

    def convert_examples_to_features(
        self,
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        training_args = None
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # TODO clean up all this to leverage built-in features of tokenizers

        # 加载lexicon抽取器
        if training_args.use_lexicon:
            from ruler.lexicon_extractor import lexicon_extractor
            lexicon_extractor = lexicon_extractor
        else:
            lexicon_extractor = None

        if training_args.use_words:
            from ruler.word_cuter import WordCuter
            index = 0
            word2id = {}
            with open(training_args.word_vocab_path, "r", encoding="utf8") as r:
                for line in r:
                    word, _ = line.split(" ")
                    word2id[word] = index
                    index += 1
            cuter = WordCuter(list(word2id.keys()))




        label_map = {label: i for i, label in enumerate(label_list)}
        if training_args.use_crf:
            pad_token_label_id = label_map["O"]

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            tokens = []
            label_ids = []
            words = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)
                if training_args.use_words:
                    # subword不利于词向量对齐
                    tmp = []
                    for t in word_tokens:
                        if t in ["[unused1]", "[UNK]"]:
                            tmp.append(t)
                        else:
                            if t.startswith("##"):
                                t = t[2:]
                            tmp.extend(list(t))
                    word_tokens = tmp

                if label == "O":
                    label = ["O"] * len(word_tokens)
                else:
                    label = [f"B-{label}"] + [f"I-{label}"]*(len(word_tokens)-1)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[l] for l in label])

            string = "".join(tokens)
            lexicons = lexicon_extractor.extract(string) if lexicon_extractor else None
            if lexicons is not None:
                lexicons = [[tokenizer.tokenize(lexicon[0]), lexicon[1], lexicon[2]] for lexicon in lexicons]
            
            words = cuter.cut(string) if training_args.use_words else None
            # print(words)
            # print(tokens)
            words_id = None
            if words is not None:
                # token对齐到word
                words_id = [word2id["[UNK]"]] * len(tokens)
                i = 0
                for word in words:
                    j = 0
                    wid = word2id.get(word, word2id["[UNK]"])
                    if word in ["[UNK]", "[unused1]"]:
                        assert tokens[i] in ["[UNK]", "[unused1]"], [tokens[i]]
                        words_id[i] = wid
                        i += 1
                        continue
                    for char in word:
                        assert char == tokens[i], [char, tokens[i]]
                        words_id[i] =  wid
                        i += 1

            # print(len(tokens))
            # print(len(words_id))
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                if words_id is not None:
                    words_id = words_id[: (max_seq_length - special_tokens_count)]


            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if words_id is not None:
                words_id += [word2id["[unused1]"]]


            position_ids = None
            lexicon_mask = None

            if lexicons is not None:
                lexicon_mask = [1] * len(tokens)
                position_ids = list(range(1, len(tokens)+1))
                for lexicon in lexicons:
                    tokens += (lexicon[0] + [sep_token])
                    position_ids += list(range(lexicon[1]+1, lexicon[1]+1+len(lexicon[0]))) + [0] # 此时还没添加[CLS]
                    label_ids += [label_map["O"]] * (len(lexicon[0])+1)
                    lexicon_mask += [0] * (len(lexicon[0])+1)

                

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
                if words_id is not None:
                    words_id += [word2id["[unused1]"]]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                if lexicon_mask is not None:
                    lexicon_mask = [1] + lexicon_mask
                if position_ids is not None:
                    position_ids = [0] + position_ids
                if words_id is not None:
                    words_id = [word2id["[unused1]"]] + words_id

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                if lexicon_mask is not None:
                    lexicon_mask = ([pad_token_label_id] * padding_length) + lexicon_mask
                if position_ids is not None:
                    position_ids = ([pad_token_label_id] * padding_length) + position_ids
                if word2id is not None:
                    word2id = ([pad_token_label_id] * padding_length) + word2id


            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                if lexicon_mask is not None:
                    lexicon_mask += [pad_token_label_id] * padding_length
                if position_ids is not None:
                    position_ids += [pad_token_label_id] * padding_length
                if words_id is not None:
                    words_id += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length, [len(input_ids), max_seq_length]
            assert len(input_mask) == max_seq_length, [len(input_mask), max_seq_length]
            assert len(segment_ids) == max_seq_length, [len(segment_ids), max_seq_length]
            assert len(label_ids) == max_seq_length, [len(label_ids), max_seq_length]
            if lexicon_mask is not None:
                assert len(lexicon_mask) == max_seq_length, [len(lexicon_mask), max_seq_length]
            if position_ids is not None:
                assert len(position_ids) == max_seq_length, [len(position_ids), max_seq_length]
            if words_id is not None:
                assert len(words_id) == max_seq_length, [len(words_id), max_seq_length]

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
                if lexicon_mask is not None:
                    logger.info("lexicon_mask: %s", " ".join([str(x) for x in lexicon_mask]))
                if position_ids is not None:
                    logger.info("position_ids: %s", " ".join([str(x) for x in position_ids]))
                if words is not None:
                    logger.info("words: %s", " ".join([str(x) for x in words]))
                if words_id is not None:
                    logger.info("words_id: %s", " ".join([str(x) for x in words_id]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids, lexicon_mask=lexicon_mask, position_ids=position_ids, words_id=words_id
                )
            )
        return features


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data.dataset import Dataset

    class TokenClassificationDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # pad_token_label_id: int = 0
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            training_args = None
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    examples = token_classification_task.read_examples_from_file(data_dir, mode)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = token_classification_task.convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                        training_args = training_args
                    )
                    logger.info(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFTokenClassificationDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = -100
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            examples = token_classification_task.read_examples_from_file(data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = token_classification_task.convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(tokenizer.padding_side == "left"),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )

            def gen():
                for ex in self.features:
                    if ex.token_type_ids is None:
                        yield (
                            {"input_ids": ex.input_ids, "attention_mask": ex.attention_mask},
                            ex.label_ids,
                        )
                    else:
                        yield (
                            {
                                "input_ids": ex.input_ids,
                                "attention_mask": ex.attention_mask,
                                "token_type_ids": ex.token_type_ids,
                            },
                            ex.label_ids,
                        )

            if "token_type_ids" not in tokenizer.model_input_names:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                    (
                        {"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])},
                        tf.TensorShape([None]),
                    ),
                )
            else:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                    (
                        {
                            "input_ids": tf.TensorShape([None]),
                            "attention_mask": tf.TensorShape([None]),
                            "token_type_ids": tf.TensorShape([None]),
                        },
                        tf.TensorShape([None]),
                    ),
                )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]
