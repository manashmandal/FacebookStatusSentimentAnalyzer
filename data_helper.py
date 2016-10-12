import pandas as pd
import os
import numpy as np
import random
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

positive_text_path = './txt_sentoken/pos/'
negative_text_path = './txt_sentoken/neg/' 

yelp_labelled = 'yelp_labelled.txt'
imdb_labelled = 'imdb_labelled.txt'
amazon_cells_labelled = 'amazon_cells_labelled.txt'
review_training_dataset = 'training.txt'

text_sentiment_columns = ['Text', 'Sentiment']

def get_data_frame(positive_file_path=positive_text_path, negative_file_path=negative_text_path):
    
    # Getting the imdb review dataset
    positive_files = [positive_text_path + pos_file_name for pos_file_name in os.listdir(positive_text_path)]
    negative_files = [negative_text_path + neg_file_name for neg_file_name in os.listdir(negative_text_path)]

    positive_texts = []
    negative_texts = []
    
    for f in positive_files:
        with open(f, 'r') as posfile:
            positive_texts.append(posfile.read())
    
    for f in negative_files:
        with open(f, 'r') as negfile:
            negative_texts.append(negfile.read())
        
    # Creating dataframes
    pos_dataframe = pd.DataFrame(positive_texts)
    neg_dataframe = pd.DataFrame(negative_texts)
    
    # Assigning positive sentiment to 1 and negative sentiment to 0
    pos_dataframe['Sentiment'] = 1
    neg_dataframe['Sentiment'] = 0

    # Adding the labels
    pos_dataframe.columns = text_sentiment_columns
    neg_dataframe.columns = text_sentiment_columns
    
    # Getting Yelp dataset
    yelp_df = pd.read_csv(yelp_labelled, delimiter='\t')
    yelp_df.columns = text_sentiment_columns
    
    # Getting imdb movie review dataset
    imdb_df = pd.read_csv(imdb_labelled, delimiter='\t')
    imdb_df.columns = text_sentiment_columns
    
    # Getting amazon product review dataset
    amazon_df = pd.read_csv(amazon_cells_labelled, delimiter='\t')
    amazon_df.columns = text_sentiment_columns
    
    # imdb review training dataset
    imdb_training_df = pd.read_csv(review_training_dataset, delimiter='\t')
    imdb_training_df.columns = text_sentiment_columns
    
    # Swapping the columns
    imdb_training_df['Sentiment'], imdb_training_df['Text'] = imdb_training_df['Text'], imdb_training_df['Sentiment']
    
    # Resetting the column labels
    imdb_df.columns = text_sentiment_columns
    
    # Merging the dataframes into one
    total_dataframe = [pos_dataframe, neg_dataframe, imdb_df, amazon_df, imdb_training_df, yelp_df]
    df = pd.concat(total_dataframe)
    
    return df



def get_shuffled_dataframe(pfp=positive_text_path, nfp=negative_text_path):
    return shuffle(get_data_frame(pfp, nfp))


def load_train_test(dataframe, split_from):
    
    # This will take the split value as percentage
    if split_from <= 1:
        split_from = int(split_from * dataframe.shape[0])
    
    df = shuffle(dataframe)
    X_train = [text for text in df.iloc[:split_from,:1]['Text']]
    y_train = [sentiment for sentiment in df.iloc[:split_from, :2]['Sentiment']]
    X_test = [text for text in df.iloc[split_from: , :1]['Text']]
    y_test = [sentiment for sentiment in df.iloc[split_from: , :2]['Sentiment']]

    # Check splitting
    print ("X_train length: {0}\n y_train length: {1}\n X_test_length: {2}\n y_test_length: {3}\n".format(len(X_train), len(y_train), len(X_test), len(y_test)))

    return ((X_train, y_train), (X_test, y_test))


def visualize_data(classifier, feature_names, n_top_features=25):
    coef = classifier.coef_.ravel()
    pos_coef = np.argsort(coef)[-n_top_features:]
    neg_coef = np.argsort(coef)[:n_top_features]
    interesting_coefs = np.hstack([neg_coef, pos_coef])
    
    # Plot them
    plt.figure(figsize=(20, 5))
    colors = ['red' if c < 0 else 'blue' for c in coef[interesting_coefs]]
    plt.bar(np.arange(2 * n_top_features), coef[interesting_coefs], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(1, 1 + 2 * n_top_features), feature_names[interesting_coefs], rotation=60, ha='right');

