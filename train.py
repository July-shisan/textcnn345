#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,3,4,4,5,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "The learning rate (default: 0.001)")
tf.flags.DEFINE_integer("top_num", 3, "Number of top K prediction classes (default: 5)")
tf.flags.DEFINE_float("threshold", 0.5, "Threshold for prediction classes (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64/128)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 50, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# add by me
tf.flags.DEFINE_integer("decay_steps", 500, "how many steps before decay learning rate. (default: 500)")
tf.flags.DEFINE_float("decay_rate", 0.95, "Rate of decay for learning rate. (default: 0.95)")
tf.flags.DEFINE_float("norm_ratio", 2, "The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess():
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels()

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print(len(x_train))
    print(len(x_dev))
    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # optimizer = tf.train.AdamOptimizer(1e-3)
            # grads_and_vars = optimizer.compute_gradients(cnn.loss)
            # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=FLAGS.learning_rate,
                                                           global_step=global_step, decay_steps=FLAGS.decay_steps,
                                                           decay_rate=FLAGS.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(cnn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=FLAGS.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            # acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
            prec_summary = tf.summary.scalar("precision-micro", cnn.precision)
            rec_summary = tf.summary.scalar("recall-micro", cnn.recall)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, prec_summary, rec_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, prec_summary, rec_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                # _, step, summaries, loss, accuracy = sess.run(
                #     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                #     feed_dict)
                _, step, summaries, loss = sess.run([train_op, global_step, train_summary_op,
                                                     cnn.loss], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if step % FLAGS.evaluate_every == 0:
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)


            def dev_step(x_validation, y_validation, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_validation,
                  cnn.input_y: y_validation,
                  cnn.dropout_keep_prob: 1.0
                }
                # step, summaries, loss, accuracy = sess.run(
                #     [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                #     feed_dict)
                step, summaries, scores, cur_loss = sess.run([global_step, dev_summary_op, cnn.scores,
                                                              cnn.loss], feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                # Predict by threshold
                predicted_labels_threshold, predicted_values_threshold = \
                    data_helpers.get_label_using_scores_by_threshold(scores=scores, threshold=FLAGS.threshold)
                # print(predicted_labels_threshold)
                # print('---------------')
                # print(predicted_values_threshold)

                cur_rec_ts, cur_acc_ts, cur_F_ts = 0.0, 0.0, 0.0

                for index, predicted_label_threshold in enumerate(predicted_labels_threshold):
                    rec_inc_ts, acc_inc_ts, F_inc_ts = data_helpers.cal_metric(predicted_label_threshold,
                                                                     y_validation[index])

                    cur_rec_ts, cur_acc_ts, cur_F_ts = cur_rec_ts + rec_inc_ts, \
                                                       cur_acc_ts + acc_inc_ts, \
                                                       cur_F_ts + F_inc_ts

                cur_rec_ts = cur_rec_ts / len(y_validation)
                cur_acc_ts = cur_acc_ts / len(y_validation)
                cur_F_ts = cur_F_ts / len(y_validation)
                print("recall {}: acc {}, F {}".format(cur_rec_ts, cur_acc_ts, cur_F_ts))

                # Predict by topK
                # topK_predicted_labels = []
                # for top_num in range(FLAGS.top_num):
                #     predicted_labels_topk, predicted_values_topk = \
                #         data_helpers.get_label_using_scores_by_topk(scores=scores, top_num=top_num + 1)
                #     topK_predicted_labels.append(predicted_labels_topk)
                #
                # cur_rec_tk = [0.0] * FLAGS.top_num
                # cur_acc_tk = [0.0] * FLAGS.top_num
                # cur_F_tk = [0.0] * FLAGS.top_num
                #
                # for top_num, predicted_labels_topK in enumerate(topK_predicted_labels):
                #     for index, predicted_label_topK in enumerate(predicted_labels_topK):
                #         rec_inc_tk, acc_inc_tk, F_inc_tk = data_helpers.cal_metric(predicted_label_topK,
                #                                                          y_validation[index])
                #         cur_rec_tk[top_num], cur_acc_tk[top_num], cur_F_tk[top_num] = \
                #             cur_rec_tk[top_num] + rec_inc_tk, \
                #             cur_acc_tk[top_num] + acc_inc_tk, \
                #             cur_F_tk[top_num] + F_inc_tk
                #
                #     cur_rec_tk[top_num] = cur_rec_tk[top_num] / len(y_validation)
                #     cur_acc_tk[top_num] = cur_acc_tk[top_num] / len(y_validation)
                #     cur_F_tk[top_num] = cur_F_tk[top_num] / len(y_validation)
                # # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                # for top_num in range(FLAGS.top_num):
                #     print("Top{0}: recall {1:g}, accuracy {2:g}, F {3:g}"
                #     .format(top_num + 1, cur_rec_tk[top_num], cur_acc_tk[top_num], cur_F_tk[top_num]))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)

                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()
