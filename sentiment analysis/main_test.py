#coding=utf-8

#################################
# 情感分析之消极言语识别，主程序模板
# file: main_test.py
#################################

import os
import pickle
import pandas as pd
from optparse import OptionParser

from utils import *

path = os.path.abspath(os.path.dirname(__file__))

###################################
# arg_parser： 读取参数列表
###################################
def arg_parser():
    oparser = OptionParser()

    oparser.add_option("-m", "--model_file", dest="model_file", help="输入模型文件 \
            must be: negative.model", default = None)

    oparser.add_option("-d", "--data_file", dest="data_file", help="输入验证集文件 \
            must be: validation_data.txt", default = None)

    oparser.add_option("-o", "--out_put", dest="out_put_file", help="输出结果文件 \
			must be: result.txt", default = None)

    (options, args) = oparser.parse_args()
    global g_MODEL_FILE
    g_MODEL_FILE = str(options.model_file)

    global g_DATA_FILE
    g_DATA_FILE = str(options.data_file)

    global g_OUT_PUT_FILE
    g_OUT_PUT_FILE = str(options.out_put_file)

###################################
# load_model： 加载模型文件
###################################
def load_model(model_file_name):
	with open(model_file_name, 'rb') as f:
		obj = pickle.load(f)
		return obj

###################################
# predict： 根据模型预测结果并输出结果文件，文件内容格式为qid\t言语\t标签
###################################
def predict(model):
	print("predict start.......")
	###################################
	# 预测逻辑和结果输出，("%d\t%s\t%d", qid, content, predict_label)
	###################################
	assert g_DATA_FILE != str(None), "data file not provided"
	assert g_OUT_PUT_FILE != str(None), "output file not provided"

	tv, classifier = model
	# X = load_dataset(g_DATA_FILE, train=False)
	df = pd.read_csv(g_DATA_FILE, sep='\t')
	X = list(df.loc[:, 'text'])
	stopwords = load_stopwords(os.path.join(path, 'cn_stopwords.txt'))
	X_test = preprocess(X, stopwords)

	X_test = tv.transform(X_test)
	result = classifier.predict(X_test)
	df['label'] = result
	df.to_csv(g_OUT_PUT_FILE, sep='\t', index=False)

	print("predict end.......")
	print('Predictions stored to', os.path.abspath(g_OUT_PUT_FILE))

###################################
# main： 主逻辑
###################################
def main():
	print("main start.....")

	assert g_MODEL_FILE != str(None), "model file not provided"
	model = load_model(g_MODEL_FILE)
	predict(model)

	print("main end.....")

if __name__ == '__main__':
	arg_parser()
	main()
