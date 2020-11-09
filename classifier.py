from preprocessing import Preprocessing
from shared.utils import isFile
from functools import lru_cache
from nltk import sent_tokenize
from utility import Utility
import contextlib
import fasttext
import logging
import os

fasttext.FastText.eprint = print
abs_path = os.path.abspath(os.path.dirname(__file__)) + "/output"

class Classifier(object):
    """ Class for predicting label in new documents
    
    :param lang_code: language model
    :param method: classification method
    :param version: model version number
    :param top_k: top k predictions
    :param min_words: min number of words for prediction
    :param clean_text: boolean flag for cleaning text
    :param min_conf_score: minimum confidence threshold
    """
    def __init__(self, lang_code, method="FastText", version="1.1", top_k=1, min_words=10, clean_text=True, min_conf_score=0.20):
        
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.top_k = top_k
        self.min_words = min_words
        self.clean_text = clean_text
        self.min_conf_score = min_conf_score
        subdir = "{}_{}_{}".format(self.lang_code, self.method, self.version)

        self.valid_langs = ["en"]
        if lang_code in self.valid_langs:
            self.filepath_model = abs_path + "/models/" + subdir + "/model.bin"  
            self.p = Preprocessing(lang_code)
            if isFile(self.filepath_model):
                self.model = self.load_model(self.filepath_model)

    @lru_cache(maxsize=128)
    def load_model(self, filepath):
        """ Load model from filepath
        
        :param filepath:
        :return: FastText model
        """
        model = None
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            model = fasttext.load_model(filepath)
        return model

    def get_metadata(self, zipped):
        """ Get metadata description in top k predictions   
        
        :param zipped: list of predictions
        :return: list

        Sample:
              Input:
                    [
                        ('__label__business', 0.8387208580970764), 
                        ('__label__politics', 0.11257829517126083), 
                        ('__label__entertainment', 0.02361930161714554), 
                        ('__label__sport', 0.015166693367064), 
                        ('__label__tech', 0.009964932687580585)
                    ]
              Output:
                    [
                        {
                            'label': 'business', 
                            'confidence': 0.8387
                        }, 
                        {
                            'label': 'politics', 
                            'confidence': 0.1126
                        }, 
                        {
                            'label': 'entertainment', 
                            'confidence': 0.0236
                        }, 
                        {
                            'label': 'sport', 
                            'confidence': 0.0152
                        }, 
                        {
                            'label': 'tech', 
                            'confidence': 0.01
                        }
                    ]      
        """
        preds = []
        for pred in zipped:
            label = Utility.filter_prefix(pred[0])
            confidence = float("{0:.4f}".format(pred[1]))
            data = {
                "label"       : label,
                "confidence"  : confidence   
            }
            preds.append(data)
        return preds

    def get_zipped_elements(self, top_k_preds):
        """ Zipp predicted labels with confidence scores

        :param top_k_preds: tuple of tuples
        :return: list of tuples

        Sample:
              Input: 
                    (
                        (
                            '__label__business', 
                            '__label__politics', 
                            '__label__entertainment', 
                            '__label__sport', 
                            '__label__tech'
                        ), 
                         array([0.83872086, 0.1125783, 0.0236193, 0.01516669, 0.00996493])
                    ) 
              Output: 
                    [
                        ('__label__business', 0.8387208580970764), 
                        ('__label__politics', 0.11257829517126083), 
                        ('__label__entertainment', 0.02361930161714554), 
                        ('__label__sport', 0.015166693367064), 
                        ('__label__tech', 0.009964932687580585)
                    ]
        """
        zipped = list(zip(top_k_preds[0], top_k_preds[1]))
        return zipped  
                        
    def get_top_k_preds(self, text, top_k):
        """ Get top k FastText predictions

        :param text: text to predict
        :param top_k: top k predictions
        :return: tuple of tuples
                first element  : predicted labels
                second element : confidence scores
        Sample:
            (
                (
                    '__label__business', 
                    '__label__politics', 
                    '__label__entertainment', 
                    '__label__sport', 
                    '__label__tech'
                ), 
                array([0.83872086, 0.1125783, 0.0236193, 0.01516669, 0.00996493])
            ) 
        """
        # get top k predictions
        top_k_preds = self.model.predict(text, k=top_k) 
        return top_k_preds

    def predict_label(self, text):
        """ Predict label for new documents
        
        :param text: text to predict
        :return: python dictionary

        Sample:
            {
                "label":"business",
                "confidence":0.8387,
                "predictions":[
                        {
                            "label":"business",
                            "confidence":0.8387
                        },
                        {
                            "label":"politics",
                            "confidence":0.1126
                        },
                        {
                            "label":"entertainment",
                            "confidence":0.0236
                        },
                        {
                            "label":"sport",
                            "confidence":0.0152
                        },
                        {
                            "label":"tech",
                            "confidence":0.01
                        }
                ],
                "message":"successful"
            }
        """
        try:
            prediction = dict()

            if text:
                if Utility.get_doc_length(text) > self.min_words:
                    if self.lang_code in self.valid_langs:
                        if self.clean_text:
                            text = self.p.text_preprocessing(text)
                    
                        text = Utility.remove_newlines(text)
                        
                        if isFile(self.filepath_model):
                            # get top k predictions
                            top_k_preds = self.get_top_k_preds(text, self.top_k)
                            # get zipped label-confidence list 
                            zipped = self.get_zipped_elements(top_k_preds)
                            # get metadata description
                            top_k_preds = self.get_metadata(zipped)

                            if top_k_preds:
                                max_conf_pred = max(top_k_preds, key=lambda k: k["confidence"])
                                max_conf_score = max_conf_pred.get("confidence")
                                
                                if max_conf_score <= self.min_conf_score:
                                    return "unknown label, confidence below threshold"

                                prediction["label"]  = max_conf_pred.get("label")
                                prediction["confidence"] = max_conf_score
                                prediction['predictions'] = top_k_preds
                                prediction['message'] = 'successful'
                            else:
                                return "no predictions found"
                        else:
                            return "model not found"
                    else:
                        return "language not supported"
                else:
                    return "required at least {} words for prediction".format(self.min_words)
            else:
                return "required textual content"
            return prediction

        except Exception:
            logging.error("exception occured", exc_info=True)


