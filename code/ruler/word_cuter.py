from ruler.flashtext import KeywordProcessor

class WordCuter(object):
    def __init__(self, words):
        self.cuter = KeywordProcessor()
        self.cuter.add_keywords_from_list(words)

    def cut(self, text):
        keywords = self.cuter.extract_keywords(text, span_info=True)
        segments = list(text)
        for word, start, end in keywords:
            segments[start] = "".join(text[start:end+1])
            for i in range(start+1, end+1):
                segments[i] = None

        segments = [s for s in segments if s is not None]
        return segments
                 


        