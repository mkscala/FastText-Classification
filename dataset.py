from sklearn.model_selection import train_test_split
from preprocessing import Preprocessing
from sklearn.utils import shuffle
from utility import Utility
import pandas as pd

class Dataset(object):
    """ Class for pre-processing dataset as Pandas DataFrame 

    :param lang_code: language text
    :param clean_text: boolean flag for cleaning textual content
    :param label_col: label column name in DataFrame
    :param text_col: text column name in DataFrame
    :param test_size: test DataFrame size
    :param random_state: set for reproducibility
    """
    def __init__(self, lang_code, clean_text=False, label_col="label", text_col="text", test_size=0.20, random_state=42):
        
        self.lang_code = lang_code
        self.clean_text = clean_text
        self.label_col = label_col
        self.text_col = text_col
        self.test_size = test_size
        self.random_state = random_state
        self.p = Preprocessing(lang_code)

    def preprocessing(self, df):
        """ Pre-process dataframe and generate train, test data 
        
        :param df: DataFrame
        """

        # get class names
        class_names = self.get_class_names(df)
 
        # convert label column to FastText format
        df[self.label_col] = df[self.label_col].apply(lambda x: '__label__'+str(x))
        
        if self.clean_text:
            # convert text column to string
            df[self.text_col] = df[self.text_col].astype(str)
            df[self.text_col] = df[self.text_col].apply(lambda x: self.p.text_preprocessing(x))
        else:
            # convert text column to string
            df[self.text_col] = df[self.text_col].astype(str)
            df[self.text_col] = df[self.text_col].apply(lambda x: Utility.remove_newlines(x))
    
        # separate input features and target
        y = df['label']
        X = df.drop('label', axis=1)
        
        # split dataframe into train, test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # concatenate training data back together
        X = pd.concat([X_train, y_train], axis=1)
        # apply sampling to fix imbalanced classes in train data
        X = self.sampling(X)     
        X = shuffle(X)
        
        # concatenate test data back together
        Y = pd.concat([X_test, y_test], axis=1)
       
        self.X = X                         # train dataset
        self.Y  = Y                        # test dataset
        self.X_test = X_test               # train text
        self.y_test = y_test               # test labels
        self.class_names = class_names     # class names
        
    def get_class_names(self, df):
        """ Get list of class names
        
        :param df: Dataframe
        :return: list
        """
        class_names = list(df[self.label_col].unique())
        return class_names
    
    def get_label_count(self, df):
        """ Get count for each label in a dictionary 
        
        :param df: DataFrame
        :return: dictionary
        Sample:
                {
                    'fake_news': 23481,
                    'true_news': 21417
                }
        """
        label_count = df[self.label_col].value_counts().to_dict()
        return label_count

    def get_avg_label_count(self, df):
        """ Get the average count across all labels in dataframe 
        
        :param df: DataFrame
        :return: int
        """
        avg_count = int( df[self.label_col].value_counts().mean())
        return avg_count

    def sampling(self, df):
        """ Sampling class-imbalance using the average records across all labels
        
        :param df: DataFrame
        :return: DataFrame
        """
        label_count = self.get_label_count(df)
        avg_count = self.get_avg_label_count(df)

        frames = []
        for label, count in label_count.items():
            if count >= avg_count:
                # downsample dataframe to the mean across all labels
                df_label = df[df[self.label_col] == label]
                df_label = df_label.head(avg_count)
                frames.append(df_label)
            else:
                # oversample dataframe to the mean across all labels
                df_label = df[df[self.label_col] == label]
                df_label = df_label.sample(avg_count, replace=True)
                frames.append(df_label)

        df = pd.concat(frames)
        return df