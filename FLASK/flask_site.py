# encoding: utf-8
# author: liaochangzeng
# e-mail: 1207338829@qq.com

import os
import sys
import json
import thulac
import numpy as nps
import tensorflow as tf
from flask import Flask, render_template
from flask import request
from random import randint
from util import Vocabulary

# load thulac and initializing
thu = thulac.thulac(seg_only=True)
thu.cut("hello world")

# load validation data
with open("data/validation.data", encoding="utf-8") as fd:
	sen_list = fd.read().strip().split("\n")

# load dictionary
with open("../CNN/data/train.data", encoding="utf-8") as fd:
    txt = fd.read().strip().split("\n")
    txt = [item.split("\t") for item in txt]
    en = [item[0] for item in txt]
    zh = [item[1] for item in txt]
vocab_en = Vocabulary(100, 5)
vocab_zh = Vocabulary(100, 5)
vocab_en.fit_transform(en)
vocab_zh.fit_transform(zh)

# load model parameters
TEXTCNN_PATH = "../CNN/runs/"
def load_model(model_root_path):
	model_dict = {}
	for file in os.listdir(model_root_path):
		_file_path = os.path.join(model_root_path, file)
		if os.path.isdir(_file_path):
			model_dict[file] = os.path.join(_file_path, "checkpoints")
	return model_dict
TEXTCNN_model = load_model(TEXTCNN_PATH)

app = Flask(__name__)

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/random')
def random_select():
	random_index = randint(0, len(sen_list)-1)
	sen = sen_list[random_index]
	sen = sen.split("\t")
	sen_en = sen[0]
	sen_zh = "".join(sen[1].split(" "))
	label = sen[2]
	if label == "1":
		label = u"相似"
	else:
		label = u"不相似"
	return json.dumps({"sen_zh": sen_zh, "sen_en": sen_en, "label": label}, ensure_ascii=False)


@app.route('/calculation', methods=['POST', 'GET'])
def calculation():
	if request.method != 'POST':
		return "Request Method Must Be POST!"

	zh_val = request.form['zh']
	en_val = request.form['en']
	model_type = request.form['model_type']
	model_parameter = request.form['model_parameter']
	zh_val = [item[0] for item in thu.cut(zh_val)]
	zh_val = " ".join(zh_val)
	zh_val = list(vocab_zh.transform([zh_val, ]))
	en_val = list(vocab_zh.transform([en_val, ]))

	model_path = "../CNN/runs/" + model_parameter + "/checkpoints/"
	with tf.Session() as sess:
		ckpt = tf.train.get_checkpoint_state(model_path)
		saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path +'.meta')
		saver.restore(sess,ckpt.model_checkpoint_path)
		graph = tf.get_default_graph()
		feed_dict = {
			graph.get_tensor_by_name("input_x_en:0"): en_val,
			graph.get_tensor_by_name("input_x_zh:0"): zh_val,
			graph.get_tensor_by_name("dropout_keep_prob:0"): 0.5,
			graph.get_tensor_by_name("is_training:0"): False
		}
		predictions = graph.get_tensor_by_name("output/predictions:0")
		predict = sess.run(predictions, feed_dict=feed_dict)
	print(predict)
	predict = predict[0]
	if predict == 1:
		result = "相似"
	else:
		result = "不相似"
	return json.dumps({"result": result})

# 获得保存模型的关键参数
@app.route('/model_parameter', methods=['POST', 'GET'])
def model_parameter():
	if request.method != 'POST':
		return "Request Method Must Be POST!"
	model_type = request.form['model_type']
	result = {}
	if model_type == "TextCNN":
		result = TEXTCNN_model
	return json.dumps(result)


app.run(debug = True)
