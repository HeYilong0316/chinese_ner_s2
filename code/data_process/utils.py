import os
from glob import glob
import pandas as pd
from ruler.flashtext import KeywordProcessor

class MentionFreq(object):   
    def __init__(self, file_name_list):
        ann_df = pd.DataFrame()
        for file_name in sorted(file_name_list):
            file_id = os.path.basename(file_name).split(".")[0]
            with open(file_name, "r", encoding="utf8") as r:
                text = r.read()
            ann = pd.read_csv(f"{file_id}.ann", sep="\t", header=None)
            ann.columns = ["id_", "label_and_span", "mention"]
            ann["file"] = file_id
            ann["label_and_span"] = ann.label_and_span.apply(
                lambda x: x.strip().split(" "))
            ann["label"] = ann.label_and_span.apply(lambda x: x[0])
            ann["span"] = ann.label_and_span.apply(
                lambda x: list(map(lambda x: int(x), x[1:])))
            ann = ann.drop("label_and_span", axis=1)
            ann["mention"] = ann.span.apply(lambda x: text[x[0]:x[1]])
            ann_df = pd.concat([ann_df, ann])

        ann_mention_df = ann_df[["mention", "label"]]
        ann_mention_df["freq"] = 1
        ann_mention_df = ann_mention_df.groupby(["mention", "label"]).agg(sum).reset_index()
        all_mention = ann_mention_df.mention.drop_duplicates().values.tolist()
        extractor = KeywordProcessor()
        extractor.add_keywords_from_list(all_mention)

        mention_total_freq = {}
        for file_name in file_name_list:
            with open(file_name, "r", encoding="utf8") as r:
                text = r.read()
            mention_list = extractor.extract_keywords(text)
            for mention in mention_list:
                mention_total_freq[mention] = mention_total_freq.get(mention, 0) + 1

        mention_total_freq = [{"mention": k, "total_freq": v} for k, v in mention_total_freq.items()]
        mention_total_freq = pd.DataFrame(mention_total_freq)
        mention_df = pd.merge(ann_mention_df, mention_total_freq, on="mention", how="left")
        mention_df["proportion"] = mention_df.freq / mention_df.total_freq
        self.mention_df = mention_df

    def get_mention_low_proportion(self, pro=0.1):
        return self.mention_df[self.mention_df.proportion<=pro].mention.tolist()

    def get_mention_high_proportion(self, pro=0.9):
        return self.mention_df[self.mention_df.proportion>=pro].mention.tolist()