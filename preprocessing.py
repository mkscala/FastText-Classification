from shared.utils import read_txt_as_list
from nltk import word_tokenize 
from utility import Utility
import os

stopwords_path = os.path.dirname(os.path.abspath("__file__")) + "/stopwords"

class Preprocessing(object):
    """ Class for pre-processing textual content

    :param lang_code: language text
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        
        self.stopwords = []
        self.valid_langs = ["en"]

        if lang_code in self.valid_langs:
            if lang_code == "en":
                self.stopwords_path = stopwords_path + "/" + lang_code
                self.stopwords = read_txt_as_list(self.stopwords_path + "/stopwords.txt")

    def remove_stopwords(self, text):
        """ Remove stopwords from text 
    
        :param text: text
        :return: filtered stopwords from text
        """
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stopwords]
        text = " ".join(filtered_tokens)
        return text

    def text_preprocessing(self, text):
        """ Filter punctuation, digits, stopwords from text

        :param text: text 
        :return: text string
        """
        text  = Utility.clean_text(text)
        text = self.remove_stopwords(text)
        return text
