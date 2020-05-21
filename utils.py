import re
import jieba
import pandas as pd

def load_dataset(path, train=True):
    df = pd.read_csv(path, sep='\t')
    X = list(df.loc[:, 'text'])
    if train:
        y = list(df.loc[:, 'label'])
        return (X, y)
    else:
        return X

def clean(sentence):
    # sentence = re.sub('[’!"#$%&\'()*+,-./:;<=>?@，。?～•ᴗﾟД≠❌●—｀Δ´' \
    # '♞ὤ̀★、…【】�⊙（）《》？“”‘’！[\\]^_`{|}~\s\n]+', '', sentence)
    # sentence = re.sub('[A-Za-z0-9]+', '', sentence)
    return ''.join(re.findall('[\u4e00-\u9fa5]+', sentence))

def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = list(f.read().split('\n'))
    return stopwords

def preprocess(X, stopwords):
    X = [clean(sentence) for sentence in X]
    X = [' '.join([w for w in jieba.lcut(sentence) if w not in stopwords]) for sentence in X]
    return X