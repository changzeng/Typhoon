# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import sys
import time
import logging
import argparse
import tensorflow as tf

import data_helpers as dh
from model_cnn import TwoLangTextCNN
from model_rnn import SiameseLSTM
from tensorflow.contrib.tensorboard.plugins import projector
from batch_loader import BatchLoader, MulBatchLoader

# Parameters
# ==================================================

# TRAIN_OR_RESTORE = input("☛ Train or Restore?(T/R) \n")
TRAIN_OR_RESTORE = "T"

while not (TRAIN_OR_RESTORE.isalpha() and TRAIN_OR_RESTORE.upper() in ['T', 'R']):
    TRAIN_OR_RESTORE = input('✘ The format of your input is illegal, please re-input: ')
logging.info('✔︎ The format of your input is legal, now loading to next step...')

TRAIN_OR_RESTORE = TRAIN_OR_RESTORE.upper()

if TRAIN_OR_RESTORE == 'T':
    logger = dh.logger_fn('tflog', 'logs/training-{0}.log'.format(time.asctime()))
if TRAIN_OR_RESTORE == 'R':
    logger = dh.logger_fn('tflog', 'logs/restore-{0}.log'.format(time.asctime()))

TRAININGSET_DIR = '../data/Train.json'
VALIDATIONSET_DIR = '../data/Validation.json'
METADATA_DIR = '../data/metadata.tsv'

# Data Parameters
tf.flags.DEFINE_string("training_data_file", TRAININGSET_DIR, "Data source for the training data.")
tf.flags.DEFINE_string("validation_data_file", VALIDATIONSET_DIR, "Data source for the validation data.")
tf.flags.DEFINE_string("metadata_file", METADATA_DIR, "Metadata file for embedding visualization"
                                                      "(Each line is a word segment in metadata_file).")
tf.flags.DEFINE_string("train_or_restore", TRAIN_OR_RESTORE, "Train or Restore.")
tf.flags.DEFINE_string("data_path", "data/train_snli.txt", "Data path.")

# Model Hyperparameters
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("pad_seq_len", 120, "Recommended padding Sequence length of data (depends on the data)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("embedding_type", 1, "The embedding type (default: 1)")
tf.flags.DEFINE_integer("fc_hidden_size", 1024, "Hidden size for fully connected layer (default: 1024)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_classes", 2, "number of your classifier")
tf.flags.DEFINE_integer("VOCAB_SIZE", 10000, "number of vocabulary")

# Training Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.")
tf.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.flags.DEFINE_integer("checkpoint_every", 50, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("gpu_options_allow_growth", True, "Allow gpu options growth")

FLAGS = tf.flags.FLAGS
dilim = '-' * 100
# FLAGS(sys.argv)
# logger.info('\n'.join([dilim, *['{0:>50}|{1:<50}'.format(attr.upper(), FLAGS.__getattr__(attr)) for attr in sorted(FLAGS.__dict__['__wrapped'])], dilim]))

parser = argparse.ArgumentParser(description="Training CNN")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
args = parser.parse_args()

mode = "rnn"

def train_cnn():
    """Training CNN model."""

    # Load sentences, labels, and training parameters
    # logger.info('✔︎ Loading data...')

    # logger.info('✔︎ Training data processing...')
    # train_data = dh.load_data_and_labels(FLAGS.training_data_file, FLAGS.embedding_dim)

    # logger.info('✔︎ Validation data processing...')
    # validation_data = dh.load_data_and_labels(FLAGS.validation_data_file, FLAGS.embedding_dim)

    # logger.info('Recommended padding Sequence length is: {0}'.format(FLAGS.pad_seq_len))

    # logger.info('✔︎ Training data padding...')
    # x_train_front, x_train_behind, y_train = dh.pad_data(train_data, FLAGS.pad_seq_len)

    # logger.info('✔︎ Validation data padding...')
    # x_validation_front, x_validation_behind, y_validation = dh.pad_data(validation_data, FLAGS.pad_seq_len)

    # Build vocabulary
    # VOCAB_SIZE = dh.load_vocab_size(FLAGS.embedding_dim)
    # pretrained_word2vec_matrix = dh.load_word2vec_matrix(VOCAB_SIZE, FLAGS.embedding_dim)
    pretrained_word2vec_matrix = None

    # Build a graph and cnn object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = FLAGS.gpu_options_allow_growth
        sess = tf.Session(config=session_conf)
        timestamp = str(int(time.time()))
        with sess.as_default():
            batch_loader = MulBatchLoader("data/train.data", FLAGS.batch_size, "runs/"+timestamp+"/")
            if mode == "cnn":
                cnn = TwoLangTextCNN(
                    sequence_length=batch_loader.max_len,
                    num_classes=FLAGS.num_classes,
                    vocab_size_en=batch_loader.vocab_size_en,
                    vocab_size_zh=batch_loader.vocab_size_zh,
                    fc_hidden_size=FLAGS.fc_hidden_size,
                    embedding_size=FLAGS.embedding_dim,
                    embedding_type=FLAGS.embedding_type,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    pretrained_embedding=pretrained_word2vec_matrix)
            elif mode == "rnn":
                cnn = SiameseLSTM(
                    sequence_length=batch_loader.max_len,
                    num_classes=FLAGS.num_classes,
                    vocab_size_en=batch_loader.vocab_size_en,
                    vocab_size_zh=batch_loader.vocab_size_zh,
                    fc_hidden_size=FLAGS.fc_hidden_size,
                    embedding_size=FLAGS.embedding_dim,
                    embedding_type=FLAGS.embedding_type,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    pretrained_embedding=pretrained_word2vec_matrix)

            # Define training procedure
            # learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate, global_step=cnn.global_step,
            #                                            decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate,
            #                                            staircase=True)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                optimizer = tf.train.AdamOptimizer(args.learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(cnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=cnn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            # grad_summaries = []
            # for g, v in zip(grads, vars):
            #     if g is not None:
            #         grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
            #         sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            #         grad_summaries.append(grad_hist_summary)
            #         grad_summaries.append(sparsity_summary)
            # grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if FLAGS.train_or_restore == 'R':
                MODEL = input("☛ Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368): ")  # The model you want to restore

                while not (MODEL.isdigit() and len(MODEL) == 10):
                    MODEL = input('✘ The format of your input is illegal, please re-input: ')
                logger.info('✔︎ The format of your input is legal, now loading to next step...')

                checkpoint_dir = 'runs/' + MODEL + '/checkpoints/'

                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))
            else:
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                logger.info("✔︎ Writing to {0}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train summaries
            # train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary, acc_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            if FLAGS.train_or_restore == 'R':
                # Load cnn model
                logger.info("✔ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                logger.info(checkpoint_file)

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Embedding visualization config
                config = projector.ProjectorConfig()
                embedding_conf = config.embeddings.add()
                embedding_conf.tensor_name = 'embedding'
                embedding_conf.metadata_path = FLAGS.metadata_file

                projector.visualize_embeddings(train_summary_writer, config)
                projector.visualize_embeddings(validation_summary_writer, config)

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, 'embedding', 'embedding.ckpt'))

            current_step = sess.run(cnn.global_step)

            def train_step(x_batch_front, x_batch_behind, y_batch, epoch):
                """A single training step"""
                feed_dict = {
                    cnn.input_x_en: x_batch_front,
                    cnn.input_x_zh: x_batch_behind,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                    cnn.is_training: True
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, cnn.global_step, train_summary_op, cnn.loss, cnn.accuracy], feed_dict)
                logger.info("epoch/step {}/{}: loss {:5.4f}, acc {:5.4f}".format(epoch+1, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(writer=None):
                """Evaluates model on a validation set"""
                total_step = 0
                total_loss = 0
                total_accuracy = 0
                total_recall = 0
                total_precision = 0
                total_f1 = 0
                total_auc = 0
                for x_batch_front, x_batch_behind, y_batch in batch_loader.gen_dev_batch():
                    feed_dict = {
                        cnn.input_x_en: x_batch_front,
                        cnn.input_x_zh: x_batch_behind,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0,
                        cnn.is_training: False
                    }
                    step, summaries, loss, accuracy, recall, precision, f1, auc = sess.run(
                        [cnn.global_step, validation_summary_op, cnn.loss, cnn.accuracy,
                         cnn.recall, cnn.precision, cnn.F1, cnn.AUC], feed_dict)
                    total_step += 1
                    total_loss += loss
                    total_accuracy += accuracy
                    total_recall += recall
                    total_precision += precision
                    total_f1 += f1
                    # total_auc += auc
                def get_div(a, b):
                    if b == 0:
                        return 0
                    return a*1.0/b
                avg_loss = get_div(total_loss, total_step)
                avg_accuracy = get_div(total_accuracy, total_step)
                avg_recall = get_div(total_recall, total_step)
                avg_precision = get_div(total_precision, total_step)
                avg_f1 = get_div(total_f1, total_step)
                # avg_auc = get_div(total_auc, total_step)
                # logger.info("total_step {0}: avg_loss {1:g}, avg_acc {2:g}, avg_recall {3:g}, avg_precision {4:g}, avg_f1 {5:g}, avg_AUC {6}"
                #             .format(total_step, avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1, avg_auc))
                logger.info("total_step {0}: avg_loss {1:g}, avg_acc {2:g}, avg_recall {3:g}, avg_precision {4:g}, avg_f1 {5:g}"
                            .format(total_step, avg_loss, avg_accuracy, avg_recall, avg_precision, avg_f1))
                avg_summaries = tf.Summary()
                loss_val = avg_summaries.value.add()
                loss_val.tag = "loss"
                loss_val.simple_value = avg_loss
                accuracy_val = avg_summaries.value.add()
                accuracy_val.tag = "accuracy"
                accuracy_val.simple_value = avg_accuracy
                if writer:
                    writer.add_summary(avg_summaries, step)

            # Training loop. For each batch...
            for epoch in range(20):
                for en_batch, zh_batch, y_batch in batch_loader.gen_batch():
                    train_step(en_batch, zh_batch, y_batch, epoch)
                    current_step = tf.train.global_step(sess, cnn.global_step)

                    if current_step % FLAGS.checkpoint_every == 0:
                        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logger.info("✔︎ Saved model checkpoint to {0}\n".format(path))

                    if current_step % FLAGS.evaluate_every == 0:
                        logger.info("\nEvaluation:")
                        validation_step(writer=validation_summary_writer)

    logger.info("✔︎ Done.")


if __name__ == '__main__':
    train_cnn()
