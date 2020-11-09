from visualization import save_confusion_matrix, save_class_distribution, save_seq_len_distribution, save_class_wordclouds
from sklearn.metrics import classification_report
from shared.utils import dump_to_json
from shared.utils import dump_to_txt
from shared.utils import make_dirs
from utility import Utility
from dataset import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import fasttext
import logging
import time
import glob
import os
import csv

pd.set_option('display.max_columns', None)
pd.set_option("max_rows", None)

class Evaluation(object):
    """ Class for generating classification model and evaluation files.
    """
    def __init__(self, lang_code, method="FastText", version="1.1", clean_text=False, label_col="label", text_col="text", test_size=0.20, 
                 random_state=42, epoch=25, lr=1.0, loss='softmax', wordNgrams=2, verbose=2, minCount=1, retrain=True, num_wordcloud_words=150):
           
        self.lang_code = lang_code
        self.method = method
        self.version = version
        self.clean_text = clean_text
        self.label_col = label_col
        self.text_col = text_col
        self.test_size = test_size
        self.random_state = random_state
        self.epoch = epoch
        self.lr = lr
        self.loss = loss
        self.wordNgrams = wordNgrams
        self.verbose = verbose
        self.minCount = minCount
        self.retrain = retrain
        self.num_wordcloud_words = num_wordcloud_words
    
    def print_results(self, N, p, r):
        """ Print classification results 
        
        :param N: num words
        :param p: precision
        :param r: recall
        """
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))

    def predicted_labels(self, model, test_texts):
        """ Get predicted labels 
        
        :param model: FastText model
        :param test_texts: DataFrame test texts
        :return: numpy predicted labels
        """
        pred_labels = [model.predict(x)[0][0] for x in test_texts]
        pred_labels = [Utility.filter_prefix(label) for label in pred_labels]
        pred_labels = np.array(pred_labels)
        return pred_labels
    
    def actual_labels(self, test_labels):
        """ Get test/actual labels 
        
        :param test_labels: DataFrame test labels
        :return: numpy test/actual labels
        """
        actual_labels = [Utility.filter_prefix(label) for label in test_labels]
        actual_labels = np.array(actual_labels)
        return actual_labels
       
    def create_model(self, df, output_path):
        """ Generate model & evaluation file to a given output path 
        
        :param df: DataFrame
        :param output_path: path to save model, dictionary, corpus, evaluation files
        """
        try:
            
            # create DataFrame copy
            orig_df = df.copy()
            
            # define output path
            subdir = "{}_{}_{}".format(self.lang_code, self.method, self.version)
            models_path = output_path + "/models/" + subdir
            eval_path = output_path + "/evaluation/" + subdir
            
            # create directories
            make_dirs(output_path)
            make_dirs(models_path)
            make_dirs(eval_path)
            make_dirs(eval_path + "/wordcloud")

            # define dataset and process df
            dataset = Dataset(
                lang_code=self.lang_code, clean_text=self.clean_text, label_col=self.label_col, 
                text_col=self.text_col, random_state=self.random_state, test_size=self.test_size
            )
            dataset.preprocessing(df)

            # save train, test to csv files
            train_file = eval_path + "/train.csv"
            test_file  = eval_path + "/test.csv"
            dataset.X.to_csv(eval_path + "/train.csv", sep="\t", index=False, header=False)
            dataset.Y.to_csv(eval_path + "/test.csv", sep="\t", index=False, header=False)

            # input train file and create FastText model
            model = fasttext.train_supervised(
                input=train_file, 
                epoch=self.epoch, 
                lr=self.lr, 
                loss=self.loss, 
                wordNgrams=self.wordNgrams, 
                verbose=self.verbose, 
                minCount=self.minCount
            )
            # apply model quantization/compression
            model.quantize(input=train_file, retrain=self.retrain)

            # get predicted labels using FastText model on test text DataFrame
            pred_labels = self.predicted_labels(model, dataset.X_test[self.text_col])
            # get actual labels from test labels DataFrame
            actual_labels = self.actual_labels(dataset.y_test)

            # format test, train labels
            dataset.X.label = dataset.X.label.apply(lambda x: Utility.filter_prefix(x))
            dataset.Y.label = dataset.Y.label.apply(lambda x: Utility.filter_prefix(x))
           
            # hyperparameters & evaluation metrics
            metrics = {
                "lang_code"     : self.lang_code,
                "method"        : self.method,
                "version"       : self.version,
                "clean_text"    : self.clean_text,
                "num_docs"      : df.shape[0],
                "label_col"     : self.label_col,
                "text_col"      : self.text_col,
                "test_size"     : self.test_size,
                "random_state"  : self.random_state,
                "epoch"         : self.epoch,
                "lr"            : self.lr,
                "wordNgrams"    : self.wordNgrams,
                "verbose"       : self.verbose,
                "minCount"      : self.minCount   
            }
            
            self.print_results(*model.test(test_file))

            # generate and save classification report
            report = classification_report(actual_labels, pred_labels)
            dump_to_txt(report, eval_path + "/classification_report.txt")
            
            # save classification model
            model.save_model(models_path + "/model.bin")
            dump_to_json(metrics, eval_path + "/evaluation.json", sort_keys=False)
            save_seq_len_distribution(df, eval_path + '/seq_length.png')
            save_class_distribution(orig_df, eval_path + '/data_dist.png')
            save_class_distribution(dataset.X, eval_path + '/train_dist.png')
            save_class_distribution(dataset.Y, eval_path + '/test_dist.png')
            save_confusion_matrix(actual_labels, pred_labels, eval_path + '/confusion_matrix.png')
            
            # save class wordclouds in train dataset
            save_class_wordclouds(dataset.X, dataset.class_names, self.num_wordcloud_words, eval_path + "/wordcloud")
        
        except Exception:
            logging.error('error occured', exc_info=True)

if __name__ == "__main__":
    df = pd.read_csv('data/dataset.csv', sep="\t")
    ev = Evaluation(lang_code="en", method="FastText", version="1.1", clean_text=True, epoch=200, lr=1.0)
    ev.create_model(df, output_path="output")