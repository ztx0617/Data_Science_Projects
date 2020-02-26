import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])
        
def upgrade(package):
    pip.main(['install', '--upgrade', package])
    
install('xgboost')
upgrade('scikit-learn')
upgrade('cloudpickle')


import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
import cloudpickle


nltk.download('stopwords')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)

    X = df['message'].values
    Y = df.iloc[:,4:].values

    category_names = df.iloc[:,4:].columns.tolist()

    return X, Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z]", " ", text.lower())
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens

def build_model():
    
    model = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0,
              learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
              nthread=None, objective='binary:logistic', random_state=2020,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
              silent=None, subsample=1, verbosity=1)
    
    pipeline = Pipeline([
        ('tfidf-vect', TfidfVectorizer(tokenizer = tokenize)),
        ('lsa', TruncatedSVD(random_state = 2020)),
        ('clf', MultiOutputClassifier(model, n_jobs = -1))
        ])
    
    parameters = {
        'tfidf-vect__max_df': [0.5, 0.75, 1.0],
        'lsa__n_components':[50, 100, 200],
        'clf__estimator__max_depth': [2, 3, 6],
        'clf__estimator__learning_rate': [0.1, 0.3, 0.5]
    }

    scorer = make_scorer(f1_score, average = 'macro', zero_division = 0)
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, cv=3)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):

    Y_predicted = model.predict(X_test)
    
    f1 = pd.Series(f1_score(Y_test, Y_predicted, average=None, zero_division = 0))
    recall = pd.Series(recall_score(Y_test, Y_predicted, average=None, zero_division = 0))
    precision = pd.Series(precision_score(Y_test, Y_predicted, average=None, zero_division = 0))
    
    metrics = pd.DataFrame({'f1': f1, 'recall': recall, 'precision': precision})
    metrics.index = category_names
    
    print(metrics)


def save_model(model, model_filepath):
    
    cloudpickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


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