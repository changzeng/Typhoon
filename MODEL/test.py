# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import numpy as np
import tensorflow as tf
import data_helpers as dh

# Parameters
# ==================================================

logger = dh.logger_fn('tflog', 'logs/test-{0}.log'.format(time.asctime()))

user_input = input("☛ Please input the subset and the model file you want to test, it should be like(11, 1490175368): ")
SUBSET = user_input.split(',')[0]
MODEL_LOG = user_input.split(',')[1][1:]

while not (SUBSET.isdigit() and int(SUBSET) in range(1, 12) and MODEL_LOG.isdigit() and len(MODEL_LOG) == 10):
    SUBSET = input('✘ The format of your input is illegal, it should be like(11, 1490175368), please re-input: ')
logger.info('✔︎ The format of your input is legal, now loading to next step...')


TRAININGSET_DIR = '../data/Model Training/Model' + SUBSET + '_Training.json'
VALIDATIONSET_DIR = '../data/Model Validation/Model' + SUBSET + '_Validation.json'
TESTSET_DIR = '../data/Model Test/Model' + SUBSET + '_Test.json'
MODEL_DIR = 'runs/' + MODEL_LOG + '/checkpoints/'
SAVE_DIR = 'results/' + MODEL_LOG

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data")
tf.flags.DEFINE_string("test_data_file", TESTSET_DIR, "Data source for the test data")
tf.flags.DEFINE_string("checkpoint_dir", MODEL_DIR, "Checkpoint directory from training run")

# Model Hyperparameters
tf.flags.DEFINE_integer("pad_seq_len", 120, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Test Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
dilim = '-' * 100
logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr))
                                for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))


def test_cnn():
    """Test CNN model."""

    # Load data
    logger.info("✔ Loading data...")
    logger.info('Recommended padding Sequence length is: {0}'.format(FLAGS.pad_seq_len))

    logger.info('✔︎ Test data processing...')
    test_data = dh.load_data_and_labels(FLAGS.test_data_file, FLAGS.embedding_dim)

    logger.info('✔︎ Test data padding...')
    x_test_front, x_test_behind, y_test = dh.pad_data(test_data, FLAGS.pad_seq_len)

    # Build vocabulary
    VOCAB_SIZE = dh.load_vocab_size(FLAGS.embedding_dim)
    pretrained_word2vec_matrix = dh.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)

    # Load cnn model
    logger.info("✔ Loading model...")
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    logger.info(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x_front = graph.get_operation_by_name("input_x_front").outputs[0]
            input_x_behind = graph.get_operation_by_name("input_x_behind").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            # pre-trained word2vec
            pretrained_embedding = graph.get_operation_by_name("embedding/embedding").outputs[0]

            # Tensors we want to evaluate
            scores = graph.get_operation_by_name("output/scores").outputs
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            softmax_scores = graph.get_operation_by_name("output/SoftMax_scores").outputs[0]
            topKPreds = graph.get_operation_by_name("output/topKPreds").outputs[0]
            accuracy = graph.get_operation_by_name("accuracy/accuracy").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            # Split the output nodes name by '|' if you have several output nodes
            output_node_names = 'output/scores|output/predictions|output/SoftMax_scores|output/topKPreds'

            # Save the .pb model file
            output_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
                                                                            output_node_names.split("|"))
            tf.train.write_graph(output_graph_def, 'graph', 'graph-cnn-{0}.pb'.format(MODEL_LOG), as_text=False)

            # Generate batches for one epoch
            batches = dh.batch_iter(list(zip(x_test_front, x_test_behind, y_test)), FLAGS.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_scores = []
            all_softmax_scores = []
            all_predictions = []
            all_topKPreds = []

            for index, x_test_batch in enumerate(batches):
                x_batch_front, x_batch_behind, y_batch = zip(*x_test_batch)
                feed_dict = {
                    input_x_front: x_batch_front,
                    input_x_behind: x_batch_behind,
                    input_y: y_batch,
                    dropout_keep_prob: 1.0,
                    is_training: False
                }
                batch_scores = sess.run(scores, feed_dict)
                all_scores = np.append(all_scores, batch_scores)

                batch_softmax_scores = sess.run(softmax_scores, feed_dict)
                all_softmax_scores = np.append(all_softmax_scores, batch_softmax_scores)

                batch_predictions = sess.run(predictions, feed_dict)
                all_predictions = np.concatenate([all_predictions, batch_predictions])

                batch_topKPreds = sess.run(topKPreds, feed_dict)
                all_topKPreds = np.append(all_topKPreds, batch_topKPreds)

                batch_loss = sess.run(loss, feed_dict)
                batch_acc = sess.run(accuracy, feed_dict)

                logger.info("✔︎ Test batch {0}: loss {1:g}, accuracy {2:g}.".format((index + 1), batch_loss, batch_acc))

            os.makedirs(SAVE_DIR)
            np.savetxt(SAVE_DIR + '/result_sub_' + SUBSET + '.txt', list(zip(all_predictions, all_topKPreds)), fmt='%s')

    logger.info("✔ Done.")


if __name__ == '__main__':
    test_cnn()
