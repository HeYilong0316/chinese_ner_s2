from ruler.flashtext import KeywordProcessor
import json
import os

lexicon_path = "../user_data/data/dicts/lexicon.json"

class LexiconExtractor(object):
    def __init__(self):
        self._dict_extractor = None
        self.setup()

    def setup(self):
        if os.path.exists(lexicon_path):
            self._dict_extractor = KeywordProcessor() 
            with open(lexicon_path, "r", encoding="utf8") as r:
                self._dict_extractor.add_keywords_from_list(json.load(r))

    def extract(self, text):
        if not self._dict_extractor:
            raise ValueError("没有实例化：_dict_extractor")
        entity_list = self._dict_extractor.extract_keywords(text, span_info=True)
        return entity_list 

lexicon_extractor = LexiconExtractor()
            

