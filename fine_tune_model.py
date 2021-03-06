# coding:utf-8

import time
import math
import numpy as np
import os.path as osp
import modeling
import tokenization
import tensorflow as tf
import optimization
from sklearn.model_selection import train_test_split
from run_classifier import create_model, file_based_input_fn_builder ,ColaProcessor,file_based_convert_examples_to_features
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve
from preprocess import count_trues, fasta2tsv, fasta2tfrecord

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('do_eval', True, 'Whether to evaluate after training')
tf.app.flags.DEFINE_boolean('do_save_model', True, 'Whether to save the model after training')
tf.app.flags.DEFINE_string('data_name', 'DPP-IV', "the name of the dataset to use")
tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
tf.app.flags.DEFINE_integer('num_train_epochs', 50, 'training epochs')
tf.app.flags.DEFINE_float('warmup_proportion', 0.1, 'proportion of warmup')
tf.app.flags.DEFINE_float('learning_rate', 2e-6, 'learning rate')
tf.app.flags.DEFINE_boolean('using_tpu', False, 'Whether to use TPU')
tf.app.flags.DEFINE_float('seq_length', 128, 'Sequence length')
tf.app.flags.DEFINE_string('data_root', './out/overlap_model', 'The location of the data set to be used')
tf.app.flags.DEFINE_string('positive_file', 'train-positive.txt', 'The name of file containing the positive trian data')
tf.app.flags.DEFINE_string('negative_file', 'train-negative.txt', 'The name of file containing the negative trian data')
tf.app.flags.DEFINE_string('kmer', '2', 'The type of word embedding')
tf.app.flags.DEFINE_string('vocab_file', './vocab/vocab_2kmer.txt', 'Dictionary location')
tf.app.flags.DEFINE_string('init_checkpoint', "./model/2kmer_model/model.ckpt", 'Initialization node of the model')
tf.app.flags.DEFINE_string('bert_config', "./bert_config_2.json", 'Bert configuration')
tf.app.flags.DEFINE_string('save_path', "./out/overlap_model/model.ckpt", 'Save location of fine-tune model')


def main():
    # The following are the input parameters.
    do_eval = FLAGS.do_eval
    do_save_model = FLAGS.do_save_model
    data_name = FLAGS.data_name
    num_train_epochs = FLAGS.num_train_epochs
    warmup_proportion = FLAGS.warmup_proportion
    learning_rate = FLAGS.learning_rate
    use_tpu = FLAGS.using_tpu
    seq_length = FLAGS.seq_length
    positive_file = FLAGS.positive_file
    negative_file=FLAGS.negative_file
    data_root = FLAGS.data_root
    kmer = FLAGS.kmer
    vocab_file = FLAGS.vocab_file
    init_checkpoint = FLAGS.init_checkpoint
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config)
    positive_file=data_root+'/'+positive_file
    negative_file=data_root+'/'+negative_file
    train_example_num,test_example_num=fasta2tfrecord(positive_file,negative_file,data_root,kmer,vocab_file)
    batch_size = FLAGS.batch_size  
    train_batch_num = math.ceil(train_example_num / batch_size)
    test_batch_num = math.ceil(test_example_num / batch_size)
    input_file = data_root + "/train.tf_record"
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75   
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    num_train_steps = int(train_example_num / batch_size * num_train_epochs)
    num_warmup_steps = int(num_train_steps * warmup_proportion)
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))   
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)   
    is_training = True
    num_labels = 2
    record={}
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, True, input_ids, input_mask, segment_ids, label_ids, num_labels, False)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    train_op = optimization.create_optimizer(total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    drop_remainder = False

    def decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: decode_record(record, name_to_features),
                batch_size=batch_size,))
        return d

    train_data = input_fn({"batch_size": batch_size})
    # Generate the training set data iterator, the iterator will output data in the loop
    iterator = train_data.make_one_shot_iterator().get_next()
    if do_eval:
        input_file = data_root + "/train.tf_record"
        dev_data = input_fn({"batch_size": batch_size})
        dev_iterator = dev_data.make_one_shot_iterator().get_next()
    val_accs = []
    sps = []
    sns = []
    if do_save_model:
        saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for step in range(num_train_epochs):
            start_time = time.time()
            for _ in range(train_batch_num):
                examples = sess.run(iterator)  # Run iterator to generate samples
                # print(examples)
                _, loss = \
                    sess.run([train_op, total_loss],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})
            print("step:", step, " loss:", round(loss, 4), end=" ")
            all_prob = []
            all_labels = []
            all_pre_labels = []
            if not do_eval:
                end_time = time.time()
                eta_time = (end_time - start_time) * \
                    (num_train_epochs - step - 1)
                print(" eta time:", eta_time, "s")
                continue
            for _ in range(test_batch_num):
                examples = sess.run(dev_iterator)
                loss, prob = \
                    sess.run([total_loss, probabilities],
                             feed_dict={input_ids: examples["input_ids"],
                                        input_mask: examples["input_mask"],
                                        segment_ids: examples["segment_ids"],
                                        label_ids: examples["label_ids"]})
                all_prob.extend(prob[:, 1].tolist())
                all_labels.extend(examples["label_ids"].tolist())
                pre_labels = np.argmax(prob, axis=-1).tolist()
                all_pre_labels.extend(pre_labels)
            acc = accuracy_score(all_labels, all_pre_labels)
            val_accs.append(acc)
            auc = roc_auc_score(all_labels, all_prob)
            mcc = matthews_corrcoef(all_labels, all_pre_labels)
            c_mat = confusion_matrix(all_labels, all_pre_labels)
            sn = c_mat[1, 1] / np.sum(c_mat[1, :])
            sp = c_mat[0, 0] / np.sum(c_mat[0, :])
            sps.append(sp)
            sns.append(sn)
            end_time = time.time()
            eta_time = (end_time - start_time) * (num_train_epochs - step - 1)
            record[step]={}
            print("SN:", sn, " SP:", sp, " ACC:", acc, " MCC:", mcc, " auROC:", auc, " eta time:",
                  eta_time, "s")
            record[step]["SN"]=sn
            record[step]["SP"]=sp
            record[step]["ACC"]=acc
            record[step]["MCC"]=mcc
            record[step]["auROC"]=auc

        if do_save_model:
            save_path = saver.save(
                sess, FLAGS.save_path)
        with open('log_2kmer.txt','w') as w:
            for rec in record:
                info=''
                for item in record[rec]:
                    info+=item+'\t{}'.format(record[rec][item])
                w.write('%s\t%s\n'%(rec,info))

main()
