# encoding: utf-8

# author: liaochangzeng
# github: https://github.com/changzeng

import os
import time
import pickle
import numpy as np
from random import randint, shuffle
from collections import defaultdict
from tensorflow.contrib import learn

class BufferWriter(object):
	def __init__(self, file_name, max_buffer_size=10*1024*1024, sep="\n"):
		self.file_name = file_name
		self.sep = sep
		self.records = []
		self.records_size = 0
		self.max_buffer_size = max_buffer_size
		self.init_file()

	def init_file(self):
		with open(self.file_name, "w+") as fd:
			pass

	def update(self, record):
		self.records.append(record)
		self.records_size += len(record)

		if self.records_size >= self.max_buffer_size:
			self.write_to_file()
			self.records = []
			self.records_size = 0

	def update_list(self, _list):
		self.update(self.sep.join(_list))

	def write_to_file(self):
		print("writting to %s" % self.file_name)
		with open(self.file_name, "a", encoding="utf-8", errors="ignore") as fd:
			fd.write(self.sep.join(self.records))
			fd.write(self.sep)

	def close(self):
		if self.records_size > 0:
			self.write_to_file()

class Shuffler:
	def __init__(self, tmp_size=5000, buffer_file_num=10):
		self.tmp_size = tmp_size
		self.buffer_file_num = buffer_file_num

	def read_file(self, fd, num):
		result = []
		for i in range(num):
			a = fd.readline().strip()
			if len(a) == 0:
				break
			result.append(a)
		return result

	def tmp_file_name(self, file_name, index):
		return file_name+".tmp_%d" % index

	def shuffle(self, file_name):
		self.shuffle_mul_file([file_name], file_name)

	def shuffle_mul_file(self, file_name_list, output_file_name):
		# create buffer writer
		tmp_writter = []
		for i in range(self.buffer_file_num):
			tmp_writter.append(BufferWriter(self.tmp_file_name(output_file_name, i), max_buffer_size=50*1024*1024, sep="\n\n"))

		for file_name in file_name_list:
			with open(file_name, "r", encoding="utf-8", errors="ignore") as fd:
				while True:
					tmp_list = self.read_file(fd, self.tmp_size)
					for item in tmp_list:
						index = randint(0, self.buffer_file_num-1)
						tmp_writter[index].update(item)
					if len(tmp_list) != self.tmp_size:
						break

		# close buffer writter
		for i in range(self.buffer_file_num):
			tmp_writter[i].close()

		order = list(range(0, self.buffer_file_num))
		shuffle(order)
		result_writter = BufferWriter(output_file_name, max_buffer_size=100*1024*1024, sep="\n\n")
		for index in order:
			tmp_file_name = self.tmp_file_name(output_file_name, index)
			with open(tmp_file_name, "r", encoding="utf-8", errors="ignore") as fd:
				while True:
					items = self.read_file(fd, self.tmp_size)
					if len(items) == 0:
						break
					result_writter.update_list(items)
					if len(items) != self.tmp_size:
						break
			# delete tmporary file
			os.remove(tmp_file_name)
		result_writter.close()

class Vocabulary(learn.preprocessing.VocabularyProcessor):
    def __init__(self, max_document_length, min_frequency=0, vocabulary=None):
        def tokenizer_fn(iterator):
            for sen in iterator:
                yield sen.split(" ")
        self.sup = super(Vocabulary, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids

class BatchLoader(object):
	def __init__(self, file_name, batch_size, model_dir):
		self.file_name = file_name
		self.data_dir = model_dir + "data/"
		for path in (model_dir, self.data_dir):
			if not os.path.exists(path):
				os.mkdir(path)
		self.batch_size = batch_size
		self.properties()

	def properties(self):
		with open(self.file_name, encoding="utf-8") as fd:
			max_a = 0
			max_b = 0
			vocab = {}
			_max = 1
			for line in fd:
				line = line.strip()
				a, b, c = line.split("\t")
				a = a.split(" ")
				b = b.split(" ")
				max_a = max(max_a, len(a))
				max_b = max(max_b, len(b))
				for word in a+b:
					if word not in vocab:
						vocab[word] = _max
						_max += 1
		self.max_len = max(max_a, max_b)
		self.vocab = vocab
		self.vocab_size = len(self.vocab) + 1

	def to_mul_ids(self, sen, _id):
		result = []
		while _id >= len(self.ids_list):
			self.ids_list.append({})
			self.ids_max.append(1)
		tmp = self.ids_list[_id]
		sen = sen[:self.max_len]
		for word in sen:
			if word not in tmp:
				tmp[word] = self.ids_max[_id]
				self.ids_max[_id] += 1
			result.append(tmp[word])
		while len(result) < self.max_len:
			result.append(0)
		return result

	def to_ids(self, vocab, sen_list):
		return vocab.transform(sen_list)

	def to_category(self, labels):
		result = np.zeros((len(labels), 2))
		for i, value in enumerate(labels):
			result[i, value] = 1
		return result

	def en_split(self, sen):
		return sen.split(" ")

	def gen_batch(self):
		# self.shuffler.shuffle_mul_file([self.file_name], self.shuffle_file)
		with open(self.file_name, encoding="utf-8") as fd:
			txt = fd.read().strip().split("\n")
		with open(self.shuffle_file, "w", encoding="utf-8") as fd:
			shuffle(txt)
			fd.write("\n".join(txt))
		with open(self.shuffle_file, encoding="utf-8") as fd:
			while True:
				x1 = []
				x2 = []
				x3 = []
				for i in range(self.batch_size):
					line = fd.readline().strip()
					if len(line) == 0:
						return
					tmp1, tmp2, tmp3 = line.split("\t")
					x1.append(self.to_ids(self.vocab, self.en_split(tmp1), self.max_len))
					x2.append(self.to_ids(self.vocab, self.en_split(tmp2), self.max_len))
					x3.append(int(tmp3))
				if len(x1) != self.batch_size or len(x2) != self.batch_size:
					break
				yield np.asarray(x1, dtype=np.int32), np.asarray(x2, dtype=np.int32), self.to_category(x3)
		# print("hello")
		# os.remove(self.shuffle_file)


class MulBatchLoader(BatchLoader):
	def __init__(self, file_name, batch_size, model_dir):
		super(MulBatchLoader, self).__init__(file_name, batch_size, model_dir)
		self.split_tarin_and_dev()

	def split_tarin_and_dev(self, _rate=0.8):
		with open(self.file_name, encoding="utf-8") as fd:
			txt = fd.read().strip().split("\n")
			shuffle(txt)
		_len = int(len(txt) * _rate)
		with open(self.data_dir+"train.data", "w" ,encoding="utf-8") as fd:
			fd.write("\n".join(txt[:_len]))
		with open(self.data_dir+"test.data", "w" ,encoding="utf-8") as fd:
			fd.write("\n".join(txt[_len:]))

	def properties(self):
		dictionary_path = "data/dictionary.data"
		if os.path.exists(dictionary_path):
			with open(dictionary_path, "rb") as fd:
				self.vocab_en = pickle.load(fd)
				self.vocab_zh = pickle.load(fd)
		else:
			with open(self.file_name, encoding="utf-8") as fd:
				txt = fd.read().strip().split("\n")
				sen_list = [sen.split("\t") for sen in txt]
			MAX_LEN = 100
			sen_list_en = [item[0] for item in sen_list]
			sen_list_zh = [item[1] for item in sen_list]
			self.vocab_en = Vocabulary(MAX_LEN, 5)
			self.vocab_zh = Vocabulary(MAX_LEN, 5)
			self.vocab_en.fit_transform(sen_list_en)
			self.vocab_zh.fit_transform(sen_list_zh)
		self.vocab_size_en = len(self.vocab_en.vocabulary_) + 1
		self.vocab_size_zh = len(self.vocab_zh.vocabulary_) + 1
		self.max_len = MAX_LEN

	def gen_batch(self):
		with open(self.data_dir+"train.data", encoding="utf-8") as fd:
			txt = fd.read().strip().split("\n")
		with open(self.data_dir+"train.data", "w", encoding="utf-8") as fd:
			shuffle(txt)
			fd.write("\n".join(txt))
		with open(self.data_dir+"train.data", encoding="utf-8") as fd:
			while True:
				en = []
				zh = []
				labels = []
				for i in range(self.batch_size):
					line = fd.readline().strip()
					if len(line) == 0:
						return
					tmp1, tmp2, tmp3 = line.split("\t")
					en.append(tmp1)
					zh.append(tmp2)
					labels.append(int(tmp3))
				if len(en) != self.batch_size or len(zh) != self.batch_size:
					return
				en = list(self.to_ids(self.vocab_en, en))
				zh = list(self.to_ids(self.vocab_zh, zh))
				yield np.asarray(en, dtype=np.int32), np.asarray(zh, dtype=np.int32), self.to_category(labels)

	def gen_dev_batch(self):
		with open(self.data_dir+"test.data", encoding="utf-8") as fd:
			txt = fd.read().strip().split("\n")
		with open(self.data_dir+"test.data", "w", encoding="utf-8") as fd:
			shuffle(txt)
			fd.write("\n".join(txt))
		with open(self.data_dir+"test.data", encoding="utf-8") as fd:
			while True:
				x1 = []
				x2 = []
				x3 = []
				for i in range(self.batch_size):
					line = fd.readline().strip()
					if len(line) == 0:
						return
					tmp1, tmp2, tmp3 = line.split("\t")
					x1.append(tmp1)
					x2.append(tmp2)
					x3.append(int(tmp3))
				if len(x1) != self.batch_size or len(x2) != self.batch_size:
					return
				x1 = list(self.to_ids(self.vocab_en, x1))
				x2 = list(self.to_ids(self.vocab_zh, x2))
				yield np.asarray(x1, dtype=np.int32), np.asarray(x2, dtype=np.int32), self.to_category(x3)


if __name__ == "__main__":
    batch_loader_sig = BatchLoader("data/train.data", 64)
    batch_loader_mul = MulBatchLoader("data/train.data", 64)
    print(batch_loader_sig.vocab_size)
    print("***************************")
    print(batch_loader_mul.vocab_size_a)
    print(batch_loader_mul.vocab_size_b)
