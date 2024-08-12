
from transformers import pipeline
from textblob import TextBlob
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to change the data types of specific columns in a DataFrame
def change_dtype_data(data):
    """
    Change the data types of specific columns in a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame whose columns need type conversion.

    Returns:
        pd.DataFrame: The DataFrame with modified data types for specific columns.
    """
    data['Author'] = data['Author'].astype('string')
    data['Date published'] = data['Date published'].astype('datetime64[ns]')
    data['Category'] = data['Category'].astype('category')
    data['Section'] = data['Section'].astype('category')
    data['Url'] = data['Url'].astype('string')
    data['Headline'] = data['Headline'].astype('string')
    data['Description'] = data['Description'].astype('string')
    data['Keywords'] = data['Keywords'].astype('string')
    data['Second headline'] = data['Second headline'].astype('string')
    data['Article text'] = data['Article text'].astype('string')
    return data

# Function to filter articles containing a specific keyword
def filter_articles_by_keyword(df, keyword):
    """
    Filter articles in a DataFrame that contain a specified keyword.

    Args:
        df (pd.DataFrame): The DataFrame containing article data.
        keyword (str): The keyword to filter the articles.

    Returns:
        pd.DataFrame: The DataFrame containing only the articles that match the keyword.
    """
    pattern = rf'\b{keyword}\b'
    filtered_df = df[df['text'].str.contains(pattern, case=False, na=False)]
    return filtered_df



def perform_textblob_sent_analysis(text):
    """
    Perform polarity analysis using TextBlob and classify the sentiment.

    Args:
        text (str): The text to be analyzed.

    Returns:
        int: 1 for positive sentiment, 0 for negative sentiment, and 0.5 for neutral sentiment.
    """
    """function to perform polarity analysis with textblob and rank the values for 

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    sentiment_list = []
    polarity = TextBlob(text).sentiment.polarity
    if polarity >= 0.34:
        return 1 # stand for positive
    elif  -0.34 >= polarity :
        return -1 # stand for negative 
    else:
        return 0 # stand for neutral sentimental
        
            
def add_textblob_sentiment_to_df(input_column):
    """
    Apply TextBlob sentiment analysis to a DataFrame column and return sentiment labels.

    Args:
        input_column (pd.Series): The column containing text data.

    Returns:
        list: A list of sentiment labels corresponding to each text entry.
    """
    sentiment_labels= []
    for article in input_column:
        sentiment = perform_textblob_sent_analysis(article)
        sentiment_labels.append(sentiment)
    return sentiment_labels

def get_df_for_business(business, data):
    """
    Filter a DataFrame by a specific business name.

    Args:
        business (str): The business name to filter by.
        data (pd.DataFrame): The DataFrame containing article data.

    Returns:
        pd.DataFrame: The DataFrame filtered by the business name.
    """
    return filter_articles_by_keyword(data, business)

def get_numb_article_business(data,my_business_list):
    """
    Count the number of articles for each business in a list.

    Args:
        data (pd.DataFrame): The DataFrame containing article data.
        my_business_list (list): A list of business names.

    Returns:
        list: A list of [number of articles, business name] pairs.
    """
    artlicle_list = []
    for business in my_business_list:
        #
        #print(business)
        length = len(filter_articles_by_keyword(data,business))
        #print(length)
        artlicle_list.append([length,business])
    return artlicle_list
        
def split_train_test(data, test_ratio):
    """
    Split a DataFrame into training and test sets.

    Args:
        data (pd.DataFrame): The DataFrame to be split.
        test_ratio (float): The proportion of data to be used as the test set.

    Returns:
        tuple: A tuple containing the training and test DataFrames.
    """
    shuffled_indicies = np.random.permutation(len(data))
    test_data_size = int(len(data) * test_ratio)
    test_indicies = shuffled_indicies[:test_data_size]
    train_indicies = shuffled_indicies[test_data_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]

def vader_sentiment(text):
    """
    Perform sentiment analysis using VADER and classify the sentiment.

    Args:
        text (str): The text to be analyzed.

    Returns:
        int: 1 for positive sentiment, -1 for negative sentiment, and 0 for neutral sentiment.
    """
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    if vs['compound'] >= 0.64:
        return 1  #stand for positive
    elif  0.33 >= vs['compound']:
        return -1 # stand for negative 
    else:
        return 0 # stand for neutral sentimental
 
def vader_df(input_column):
    """
    Apply VADER sentiment analysis to a DataFrame column and return sentiment labels.

    Args:
        input_column (pd.Series): The column containing text data.

    Returns:
        list: A list of sentiment labels corresponding to each text entry.
    """
    sentiment_labels= []
    for article in input_column:
        sentiment = vader_sentiment(article)
        sentiment_labels.append(sentiment)
    return sentiment_labels

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    """
    Preprocess text data by removing punctuation, tokenizing, and removing stopwords.

    Args:
        text (str): The text to be preprocessed.

    Returns:
        str: The preprocessed text.
    """
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def senti_log_ress_multiple_datasets(train_datasets, test_datasets, labels, test_labels, vectorizer):
    """
    Applies TF-IDF vectorization, trains Logistic Regression models, and makes predictions on multiple datasets.

    Parameters:
    train_datasets (list of pd.Series or list of list of str): List of training text datasets.
    test_datasets (list of pd.Series or list of list of str): List of test text datasets.
    labels (list of pd.Series or list of list of int): List of training labels.
    test_labels (list of pd.Series or list of list of int): List of test labels.

    Returns:
    tuple: A tuple containing the list of trained models, vectorized training datasets, vectorized test datasets,
           the list of TF-IDF vectorizers, the evaluation reports, and the predictions added to the test datasets.
    """
    vectorized_train_datasets = []
    vectorized_test_datasets = []
    vectorizers = []
    models = []
    evaluation_reports = []
    test_datasets_with_predictions = []

    for X_train, X_test, y_train, y_test in zip(train_datasets, test_datasets, labels, test_labels):
    #for i, (X_train, X_test, y_train, y_test) in enumerate(zip(train_datasets, test_datasets, labels, test_labels)):
        #print(f"Evaluating dataset {i+1} with min_df={min_df} and max_df={max_df}")
        # Initialize and fit the TF-IDF vectorizer
        #vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        vectorizers.append(vectorizer)
        vectorized_train_datasets.append(X_train_tfidf)
        vectorized_test_datasets.append(X_test_tfidf)

        # Initialize and train the logistic regression model
        model = LogisticRegression()
        model.fit(X_train_tfidf, y_train)
        models.append(model)

        # Make predictions and evaluate the model
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        evaluation_reports.append((accuracy, report,conf_matrix))

        # Add predictions to the test dataset
        X_test_with_predictions = pd.DataFrame(X_test, columns=['text'])
        X_test_with_predictions['predicted_label'] = y_pred
        test_datasets_with_predictions.append(X_test_with_predictions)

    return models, vectorized_train_datasets, vectorized_test_datasets, vectorizers, evaluation_reports, test_datasets_with_predictions


def calculate_classification_report(input_classification_report):
    """
    Calculate and print classification reports for a list of true and predicted label pairs.

    Args:
        input_classification_report (list of tuples): A list where each element is a tuple containing
                                                      the true labels and predicted labels as two arrays.

    Returns:
        None: This function prints the classification reports for each pair of true and predicted labels.
    """
    for y_true_pred_pair in input_classification_report:
        print(classification_report(y_true_pred_pair[0],y_true_pred_pair[1]))
        