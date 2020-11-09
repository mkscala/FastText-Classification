from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.rcParams['figure.figsize'] = (10, 5)

def save_confusion_matrix(actual_labels, pred_labels, output_path):
    """ Save plot confusion matrix using actual, predicted labels to a given output path  
        
    :param actual_labels: list
    :param pred_labels: list
    :param output_path: output path
    """
    df = pd.DataFrame({'actual_labels': actual_labels, 'pred_labels': pred_labels}, columns=['actual_labels', 'pred_labels'])
    cm_df = pd.crosstab(df['actual_labels'], df['pred_labels'], rownames=['Actual'], colnames=['Predicted'], margins=True)
    plt.figure()
    ax = sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.savefig(output_path)
    plt.clf()
    
def save_class_distribution(df, output_path):
    """ Save plot class distribution found in a dataset
        
    :param df: DataFrame
    :param output_path: output path
    """
    plt.figure()
    ax = sns.countplot(df['label'])
    plt.xlabel('label')
    plt.savefig(output_path)
    plt.clf()

def save_seq_len_distribution(df, output_path):
    """ Save plot text length distribution found in a dataset
        
    :param df: DataFrame
    :param output_path: output path
    """
    plt.figure()
    sentences = df['text'].tolist()
    seq_len = [len(sentence.split()) for sentence in sentences]
    pd.Series(seq_len).hist(bins=30)
    plt.savefig(output_path)
    plt.clf()

def save_class_wordclouds(df, class_names, num_wordcloud_words, output_path):
    """ Save topics visualization to a given output path 
    
    :param df: original DataFrame
    :param class_names: list of labels/classes
    :param num_wordcloud_words: generated dictionary
    :param output_path: path to save visualization
    """
    # loop through all class names
    for c in class_names:
        
        # subset dataframe by label
        df_label = df[df['label'] == c]

        # loop through dataframe text column and get all sentences
        sentences = []
        for sentence in df_label.text:
            sentences.append(sentence)
        
        # concatenate sentences on space delimiter
        text = pd.Series(sentences).str.cat(sep=' ')
        # generate wordcloud from newly generated text
        wordcloud = WordCloud(background_color="white").generate(text)
        plt.figure()
        plt.axis('off')
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.savefig(output_path + "/" + str(c))
        plt.clf()
