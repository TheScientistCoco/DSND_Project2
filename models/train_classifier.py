# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

# import statements
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# import ML modules
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(database_filepath):
    """
    Load data from database
    
    Arguments:
        database_filepath -> Path to SQLite destination database
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponseMaster', engine)
    X = df.message
    Y = df.iloc[:,4:]

    # listing the columns
    category_names = list(np.array(Y.columns))

    return X, Y, category_names


def tokenize(text):
    """
    Tokenize the text
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the provided text
    """
    # url regular expression and english stop words
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    stop_words = stopwords.words("english")
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # remove punctuation characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            # lemmatize, normalize case, and remove leading/trailing white space
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build pipeline for message classification, the parameters for the pipeline are obtained from GridSearchCV
    
    Output:
        A Scikit ML Pipeline that process text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 3],
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, n_jobs = 4, verbose = 2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the classificaiton model
    
    This function applies a ML pipeline to a test set and prints out the model performance (accuracy, F1 score, precision and recall)
    
    Arguments:
        model -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    Y_pred = model.predict(X_test)
    # print the overall accuracy of the model
    accuracy = (Y_pred == Y_test).values.mean()
    print ('Model Overall Accuracy: {}'.format(accuracy))
    
    # print the F1 score, precision and recall for each category
    print (classification_report(Y_test.values, Y_pred, target_names = category_names))

def save_model(model, model_filepath):
    """
    Save the classification model
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Arguments:
        model -> GridSearchCV or Scikit Pipeline object
        pickle_filepath -> destination path to save .pkl file
    
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()