import os
import pickle
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from utils import *

path = os.path.abspath(os.path.dirname(__file__))

def save_model(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', action='store_true', help='print classification result')
    parser.add_argument('--save', help='model saving path', required=False, type=str, 
        default=os.path.join(path, 'negative.model'))
    args = parser.parse_args()

    X, y = load_dataset(os.path.join(path, 'train_data.txt'), train=True)
    stopwords = load_stopwords(os.path.join(path, 'cn_stopwords.txt'))
    X = preprocess(X, stopwords)

    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, min_df=2, max_df=4)
    X_train = tv.fit_transform(X)

    model = LogisticRegression(class_weight='balanced', solver='newton-cg', C=10)
    model.fit(X_train, y)
    result = model.predict(X_train)

    if args.print:
        print(accuracy_score(y, result))
        print(classification_report(y, result))

    if args.save:
        obj = (tv, model)
        save_model(obj, args.save)

if __name__ == '__main__':
    main()