# encoding: utf-8

# author: liaochangzeng
# github: https://github.com/changzeng

import tensorflow as tf
import numpy as np

def linear(input_, output_size, scope=None):
    """
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        input_: a tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
    """

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: {0}".format(str(shape)))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: {0}".format(str(shape)))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        W = tf.get_variable("W", [output_size, input_size], dtype=input_.dtype)
        b = tf.get_variable("b", [output_size], dtype=input_.dtype)

    return tf.nn.xw_plus_b(input_, tf.transpose(W), b)

def highway(input_, size, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """
    Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(linear(input_, size, scope=('highway_lin_{0}'.format(idx))))
            t = tf.sigmoid(linear(input_, size, scope=('highway_gate_{0}'.format(idx))) + bias)
            output = t * g + (1. - t) * input_
            input_ = output

    return output

class SiameseLSTM(object):
    """
    A LSTM based deep Siamese network for text similarity.
    Uses an word embedding layer (looks up in pre-trained w2v), followed by a biLSTM and Energy Loss layer.
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size_en, vocab_size_zh, fc_hidden_size, embedding_size,
            embedding_type, l2_reg_lambda=0.0, pretrained_embedding=None):

        # Placeholders for input, output, dropout_prob and training_tag
        self.input_x_en = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_en")
        self.input_x_zh = tf.placeholder(tf.int32, [None, sequence_length], name="input_x_zh")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        hidden_units = 500
        self.hidden_units = hidden_units
        self.fc_hidden_size = fc_hidden_size
        self.num_classes = num_classes

        # Embedding Layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Use random generated the word vector by default
            # Can also be obtained through our own word vectors trained by our corpus
            if pretrained_embedding is None:
                self.embedding_front = tf.Variable(tf.random_uniform([vocab_size_en, embedding_size], -1.0, 1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding_a")

                self.embedding_behind = tf.Variable(tf.random_uniform([vocab_size_zh, embedding_size], -1.0, 1.0,
                                                               dtype=tf.float32), trainable=True, name="embedding_b")
            else:
                if embedding_type == 0:
                    self.embedding = tf.constant(pretrained_embedding, dtype=tf.float32, name="embedding")
                if embedding_type == 1:
                    self.embedding = tf.Variable(pretrained_embedding, trainable=True,
                                                 dtype=tf.float32, name="embedding")
            self.embedded_sentence_front = tf.nn.embedding_lookup(self.embedding_front, self.input_x_en)
            self.embedded_sentence_behind = tf.nn.embedding_lookup(self.embedding_behind, self.input_x_zh)

        with tf.name_scope("output"):
            self.out1=self.stackedRNN(self.embedded_sentence_front, self.dropout_keep_prob, "side1", embedding_size, sequence_length, hidden_units)
            self.out2=self.stackedRNN(self.embedded_sentence_behind, self.dropout_keep_prob, "side2", embedding_size, sequence_length, hidden_units)
            self.combine = tf.concat([self.out1, self.out2], 1)

        # Fully Connected Layer
        with tf.name_scope("fc"):
            W = tf.Variable(tf.truncated_normal(shape=[self.hidden_units*2, self.fc_hidden_size], stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.fc_hidden_size], dtype=tf.float32), name="b")
            self.fc = tf.nn.xw_plus_b(self.combine, W, b)

            # Batch Normalization Layer
            self.fc_bn = tf.layers.batch_normalization(self.fc, training=True)

            # Apply nonlinearity
            self.fc_out = tf.nn.relu(self.fc_bn, name="relu")

        # Highway Layer
        self.highway = highway(self.fc_out, self.fc_out.get_shape()[1], num_layers=1, bias=0, scope="Highway")

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.highway, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, self.num_classes], stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes], dtype=tf.float32), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="logits")
            self.softmax_scores = tf.nn.softmax(self.logits, name="softmax_scores")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

                # Calculate mean cross-entropy loss, L2 loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            losses = tf.reduce_mean(losses, name="softmax_losses")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add(losses, l2_losses, name="loss")

        # Number of correct predictions
        with tf.name_scope("num_correct"):
            correct = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.num_correct = tf.reduce_sum(tf.cast(correct, "float"), name="num_correct")

        # Calculate Fp
        with tf.name_scope("fp"):
            fp = tf.metrics.false_positives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fp = tf.reduce_sum(tf.cast(fp, "float"), name="fp")

        # Calculate Fn
        with tf.name_scope("fn"):
            fn = tf.metrics.false_negatives(labels=tf.argmax(self.input_y, 1), predictions=self.predictions)
            self.fn = tf.reduce_sum(tf.cast(fn, "float"), name="fn")

        # Calculate Recall
        with tf.name_scope("recall"):
            self.recall = self.num_correct / (self.num_correct + self.fn)

        # Calculate Precision
        with tf.name_scope("precision"):
            self.precision = self.num_correct / (self.num_correct + self.fp)

        # Calculate F1
        with tf.name_scope("F1"):
            self.F1 = (2 * self.precision * self.recall) / (self.precision + self.recall)

        # Calculate AUC
        with tf.name_scope("AUC"):
            self.AUC = tf.metrics.auc(self.softmax_scores, self.input_y, name="AUC")


    def stackedRNN(self, x, dropout, scope, embedding_size, sequence_length, hidden_units):
        n_hidden=hidden_units
        n_layers=1
        # Prepare data shape to match `static_rnn` function requirements
        x = tf.unstack(tf.transpose(x, perm=[1, 0, 2]))
        # print(x)
        # Define lstm cells with tensorflow
        # Forward direction cell

        with tf.name_scope("fw"+scope),tf.variable_scope("fw"+scope):
            stacked_rnn_fw = []
            for _ in range(n_layers):
                fw_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell,output_keep_prob=dropout)
                stacked_rnn_fw.append(lstm_fw_cell)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)

            outputs, _ = tf.nn.static_rnn(lstm_fw_cell_m, x, dtype=tf.float32)
        return outputs[-1]

    def contrastive_loss(self, y,d,batch_size):
        tmp= y *tf.square(d)
        #tmp= tf.mul(y,tf.square(d))
        tmp2 = (1-y) *tf.square(tf.maximum((1 - d),0))
        return tf.reduce_sum(tmp +tmp2)/batch_size/2
