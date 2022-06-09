from sklearn.model_selection import train_test_split
import time
import math
import numpy as np
import os.path as osp
import modeling
import tokenization
import tensorflow as tf
import optimization
from run_classifier import create_model, file_based_input_fn_builder ,ColaProcessor,file_based_convert_examples_to_features
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, \
    confusion_matrix, roc_curve

def count_trues(pre_labels, true_labels):
    shape = true_labels.shape
    zeros = np.zeros(shape=shape)
    ones = np.ones(shape=shape)
    pos_example_index = (true_labels == ones)
    neg_example_index = (true_labels == zeros)
    right_example_index = (pre_labels == true_labels)
    true_pos_examples = np.sum(np.logical_and(pos_example_index, right_example_index))
    true_neg_examples = np.sum(np.logical_and(neg_example_index, right_example_index))
    return np.sum(pos_example_index), np.sum(neg_example_index), true_pos_examples, true_neg_examples

def fasta2tsv(train_positive,train_negative,kmer):
    train_rec_p=[]
    train_rec_n=[]
    with open(train_positive, 'r') as f1:
        with open(train_negative,'r') as f2:
            for line in f1.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '>' :
                    continue
                seq=''
                tmp = line
                length = len(tmp)
                for i in range(0, length, kmer):
                    if length - i >= kmer:
                        seq += line[i:i+kmer] + " "
                    else:
                        seq += line[i:] + " "
                res ="train\t1\t\t"+seq
                train_rec_p.append(res)
            train_p_set, train_p_dev = train_test_split(train_rec_p, test_size=0.2,random_state=109)
            for line in f2.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '>':
                    continue
                seq=''
                tmp = line
                length = len(tmp)
                for i in range(0, length, kmer):
                    if length - i >= kmer:
                        seq += line[i:i+kmer] + " "
                    else:
                        seq += line[i:]+ " "
                res = "train\t0\t\t" + seq
                train_rec_n.append(res)
            train_n_set, train_n_dev = train_test_split(train_rec_n, test_size=0.1,random_state=109)
    train_all_set=train_p_set+train_n_set+train_p_dev+train_n_dev
    dev_all_set=[]
    return train_all_set,dev_all_set

def fasta2tfrecord(train_positive,train_negative,path,kmer, vocab):
    vocab_file = vocab
    kmer=int(kmer)
    train_all_set,dev_all_set=fasta2tsv(train_positive,train_negative,kmer)
    train_seq_num=len(train_all_set)
    valid_seq_num=train_seq_num
    out_put_path=path+'/'
    if osp.exists(out_put_path)==False:
        os.mkdir(out_put_path)
    out_put_train=out_put_path+'train.tsv'
    out_put_dev=out_put_path+'dev.tsv'
    with open(out_put_train, 'w') as w1:
        with open(out_put_dev, 'w') as w2:
            for i,rec in enumerate(train_all_set):
                if i ==len(train_all_set)-1:
                    w1.writelines(rec)
                else:
                    w1.writelines('%s\n'%(rec))
            for i,rec in enumerate(dev_all_set):
                if i ==len(dev_all_set)-1:
                    w2.writelines(rec)
                else:
                    w2.writelines('%s\n'%(rec))
    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.get_train_examples(out_put_path)
    train_file = out_put_path+ "train.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(examples, label_list, 128, tokenizer, train_file)
    examples = processor.get_dev_examples(out_put_path)
    valid_file = out_put_path + "dev.tf_record"
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(examples, label_list, 128, tokenizer, valid_file)
    return train_seq_num,valid_seq_num