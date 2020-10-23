from ruler.flashtext import KeywordProcessor


class DictExtractor(object):
    def __init__(self):
        self._dict_extractor = KeywordProcessor()

    def from_dict_file(self, dict_file):
        import pandas as pd
        datas = pd.read_csv(dict_file)
        entity_dict = datas.groupby("label").agg(lambda x: list(x)).T.to_dict(orient="records")[0]
        self._dict_extractor.add_keywords_from_dict(entity_dict)

    def extract(self, text):
        entity_list = self._dict_extractor.extract_keywords(text, span_info=True)
        return entity_list 

dict_extractor = DictExtractor()
dict_extractor.from_dict_file("datas/dicts_and_pattern/flashtext_dict.csv")
            

