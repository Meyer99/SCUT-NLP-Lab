import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from utils import *

path = os.path.abspath(os.path.dirname(__file__))

def train_logistic_regression(X, y):
    from sklearn.linear_model import LogisticRegression
    param_grid = {}
    param_grid['C'] = [0.05, 0.1, 5, 10, 15]
    model = LogisticRegression(class_weight='balanced', solver='newton-cg')
    kfold = StratifiedKFold(n_splits=5, random_state=2020)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(X=X, y=y)
    print('Best:%s Used Param%s'%(grid_result.best_score_, grid_result.best_params_))

def train_svm(X, y):
    from sklearn.svm import SVC
    param_grid = {}
    param_grid['C'] = [0.1]
    model = SVC(class_weight='balanced')
    kfold = StratifiedKFold(n_splits=5, random_state=2020)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(X=X, y=y)
    print('Best:%s Used Param%s'%(grid_result.best_score_, grid_result.best_params_))

def train_multinomial_nb(X, y):
    from sklearn.naive_bayes import MultinomialNB
    param_grid = {}
    param_grid['alpha'] = [0.001, 0.01, 0.1, 1.5, 2, 2.5, 5, 10, 15, 25, 50, 75, 100]
    model = MultinomialNB()
    kfold = StratifiedKFold(n_splits=5, random_state=2020)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(X=X, y=y)
    print('Best:%s Used Param%s'%(grid_result.best_score_, grid_result.best_params_))

def train_random_forest_classifier(X, y):
    from sklearn.ensemble import RandomForestClassifier
    param_grid = {}
    param_grid['n_estimators'] = [10]
    model = RandomForestClassifier(class_weight='balanced')
    kfold = StratifiedKFold(n_splits=5, random_state=2020)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kfold)
    grid_result = grid.fit(X=X, y=y)
    print('Best:%s Used Param%s'%(grid_result.best_score_, grid_result.best_params_))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='logistic_regression', help='training method', 
        choices=['logistic_regression', 'svm', 'random_forest', 'multinomial_nb'])
    args = parser.parse_args()

    # Load dataset and stopwords, preprocess the data so that it can fit into vectorizer
    X, y = load_dataset(os.path.join(path, 'train_data.txt'), train=True)
    stopwords = load_stopwords(os.path.join(path, 'cn_stopwords.txt'))
    X = preprocess(X, stopwords)

    if args.method == 'multinomial_nb':
        cv = CountVectorizer()
        X_train = cv.fit_transform(X)
        train_multinomial_nb(X_train, y)
    else:
        # Create a TfidfVectorizer and transform the training data
        tv = TfidfVectorizer(use_idf=True, smooth_idf=True, min_df=2, max_df=4)
        X_train = tv.fit_transform(X)

        if args.method == 'logistic_regression':
            train_logistic_regression(X_train, y) 
        if args.method == 'svm':
            train_svm(X_train, y)
        if args.method == 'random_forest':
            train_random_forest_classifier(X_train, y)

if __name__ == '__main__':
    main()



