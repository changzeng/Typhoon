# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import multiprocessing
import gensim
import logging
import json
import numpy as np

from pylab import *
from gensim.models import word2vec

TEXT_DIR = '../data/content.txt'
METADATA_DIR = '../data/metadata.tsv'


def logger_fn(name, file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_metadata_file(vocab_size, embedding_size, output_file=METADATA_DIR):
    """
    Create the metadata file based on the corpus file(Use for the Embedding Visualization later).
    :param input_file: The corpus file
    :param output_file: The metadata file (default: 'metadata.tsv')
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        model = gensim.models.Word2Vec.load(word2vec_file)
        word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

        with open(output_file, 'w+') as fout:
            for word in word2idx_sorted:
                if word[0] is None:
                    logging.info("Emply Line, should replecaed by any thing else, or will cause a bug of tensorboard")
                    fout.write('<Empty Line>' + '\n')
                else:
                    fout.write(word[0] + '\n')
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    """
    Create the word2vec model based on the given embedding size and the corpus file.
    :param embedding_size: The embedding size
    :param input_file: The corpus file
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        logging.info('☛ The word2vec model you want create already exists!')
    else:
        sentences = word2vec.LineSentence(input_file)
        # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
        model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                       sg=0, workers=multiprocessing.cpu_count())
        model.save(word2vec_file)


def load_vocab_size(embedding_size):
    """
    Return the vocab size of the word2vec file.
    :param embedding_size: The embedding size
    :return: The vocab size of the word2vec file
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        model = word2vec.Word2Vec.load(word2vec_file)
        return len(model.wv.vocab.items())
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


def data_word2vec(input_file, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Returns the class Data(includes the data tokenindex and data labels).
    :param input_file: The research data
    :param word2vec_model: The word2vec model file
    :return: The class Data(includes the data tokenindex and data labels)
    """
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def token_to_index(content):
        result = []
        for item in content:
            id = vocab.get(item)
            if id is None:
                id = 0
            result.append(id)
        return result

    if input_file.endswith('.json'):
        with open(input_file) as fin:
            labels = []
            front_content_indexlist = []
            behind_content_indexlist = []
            for index, eachline in enumerate(fin):
                data = json.loads(eachline)
                labels.append(data['label'])
                front_content_indexlist.append(token_to_index(data['front_features']))
                behind_content_indexlist.append(token_to_index(data['behind_features']))
            total_line = index + 1

        class Data:
            def __init__(self):
                pass

            @property
            def number(self):
                return total_line

            @property
            def front_tokenindex(self):
                return front_content_indexlist

            @property
            def behind_tokenindex(self):
                return behind_content_indexlist

            @property
            def labels(self):
                return labels

        return Data()
    else:
        logging.info('✘ The research data is not a json file. '
                     'Please preprocess the research data into the json file.')


def load_word2vec_matrix(vocab_size, embedding_size):
    """
    Return the word2vec model matrix.
    :param vocab_size: The vocab size of the word2vec model file
    :param embedding_size: The embedding size
    :return: The word2vec model matrix
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if os.path.isfile(word2vec_file):
        model = gensim.models.Word2Vec.load(word2vec_file)
        vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
        vector = np.zeros([vocab_size, embedding_size])
        for key, value in vocab.items():
            if len(key) > 0:
                vector[value] = model[key]
        return vector
    else:
        logging.info("✘ The word2vec file doesn't exist. "
                     "Please use function <create_vocab_size(embedding_size)> to create it!")


def load_data_and_labels(data_file, embedding_size):
    """
    Loads research data from files, splits the data into words and generates labels.
    Returns split sentences, labels and the max sentence length of the research data.
    :param data_file: The research data
    :param embedding_size: The embedding size
    :returns: The class data and the max sentence length of the research data
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if os.path.isfile(word2vec_file):
        model = word2vec.Word2Vec.load(word2vec_file)
    else:
        create_word2vec_model(embedding_size, TEXT_DIR)

    # Load data from files and split by words
    data = data_word2vec(input_file=data_file, word2vec_model=model)

    # plot_seq_len(data_file, data)

    logging.info('Found {0} texts.'.format(data.number))

    return data


def pad_sequences(data, maxlen, value):
    result = [[value]*maxlen for i in range(len(data))]
    result = np.asarray(result)
    for i, fea in enumerate(data):
        for j, value in enumerate(fea):
            reault[i, j] = value
    return result


def to_categorical(labels, nb_classes):
    result = np.zeros((len(labels), nb_classes))
    for i, value in enumerate(labels):
        result[i, value] = 1
    return result


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Returns the padded data and data labels.
    :param data: The research data
    :param max_seq_len: The max sentence length of research data
    :returns: The padded data and data labels
    """
    data_front = pad_sequences(data.front_tokenindex, maxlen=pad_seq_len, value=0.)
    data_behind = pad_sequences(data.behind_tokenindex, maxlen=pad_seq_len, value=0.)
    labels = to_categorical(data.labels, nb_classes=2)
    return data_front, data_behind, labels


def plot_seq_len(data_file, data, percentage=0.98):
    """
    Visualizing the sentence length of each data sentence.
    :param data_file: The data_file
    :param data: The class Data (includes the data tokenindex and data labels)
    :param percentage: The percentage of the total data you want to show
    """
    data_analysis_dir = '../data/data_analysis/'
    if 'train' in data_file.lower():
        output_file = data_analysis_dir + 'Train Sequence Length Distribution Histogram.png'
    if 'validation' in data_file.lower():
        output_file = data_analysis_dir + 'Validation Sequence Length Distribution Histogram.png'
    if 'test' in data_file.lower():
        output_file = data_analysis_dir + 'Test Sequence Length Distribution Histogram.png'
    result = dict()
    for x in (data.front_tokenindex + data.behind_tokenindex):
        if len(x) not in result.keys():
            result[len(x)] = 1
        else:
            result[len(x)] += 1
    freq_seq = [(key, result[key]) for key in sorted(result.keys())]
    x = []
    y = []
    avg = 0
    count = 0
    border_index = []
    for item in freq_seq:
        x.append(item[0])
        y.append(item[1])
        avg += item[0] * item[1]
        count += item[1]
        if (count / 2) > data.number * percentage:
            border_index.append(item[0])
    avg = avg / (2 * data.number)
    logging.info('The average of the data sequence length is {0}'.format(avg))
    logging.info('The recommend of padding sequence length should more than {0}'.format(border_index[0]))
    xlim(0, 200)
    plt.bar(x, y)
    plt.savefig(output_file)
    plt.close()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。
    Generate a  batch iterator for a dataset.
    :param data: The data
    :param batch_size: The size of the data batch
    :param num_epochs: The number of epoches
    :param shuffle: Shuffle or not
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
