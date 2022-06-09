# CODing:utf-8
import numpy as np
import modeling
import tokenization
import tensorflow as tf
import math
from run_classifier import create_model, file_based_input_fn_builder, ColaProcessor, file_based_convert_examples_to_features


def fasta2record(input_file, output_file, vocab_file, kmer=1):
    # This function gets an input_file which is .fasta
    # This function returns the numbers of sequences in input_file
    # This function will check if the input_file is right
    with open(input_file) as f:
        lines = f.readlines()
    for index, line in enumerate(lines):
        if index % 2 == 0:
            if line[0] != ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
        else:
            if line[0] == ">":
                print("Row " + str(index + 1) + " is wrong!")
                exit()
    seq_num = int(len(lines) / 2)
    with open("temp.tsv", "w") as f:
        for line in lines:
            if line[0] != ">":
                seq = ""
                line= line.strip()
                length = len(line)
             
                for i in range(0, length, kmer):
                    if length - i >= kmer:
                        seq += line[i:i+kmer] + " "
                    else:
                        seq += line[i:] + " "
                seq += "\n"
                f.write("train\t1\t\t" + seq)
    processor = ColaProcessor()
    label_list = processor.get_labels()
    examples = processor.ljy_get_dev_examples("temp.tsv")
    print(examples)
    train_file = "predict.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    file_based_convert_examples_to_features(
        examples, label_list, 128, tokenizer, train_file)
    return seq_num


def main(data_name, out_file, model_path, kmer=1, config_file="./bert_config_1.json",
         vocab_file="./vocab/vocab_1kmer.txt"):
    use_tpu = False
    batch_size = 32
    seq_length = 128
    init_checkpoint = model_path
    bert_config = modeling.BertConfig.from_json_file(config_file)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.75
    samples_num = fasta2record(data_name, "predict.tf_record", vocab_file, kmer=kmer)
    batch_num = math.ceil(samples_num / batch_size)
    input_file = "predict.tf_record"
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    input_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    input_mask = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    segment_ids = tf.placeholder(dtype=tf.int32, shape=(None, 128))
    label_ids = tf.placeholder(dtype=tf.int32, shape=(None,))
    is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
    num_labels = 2
    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, False, input_ids, input_mask, segment_ids, label_ids,
        num_labels, False)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    (assignment_map, initialized_variable_names
    ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
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
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: decode_record(record, name_to_features),
                batch_size=batch_size,))
        return d

    predict_data = input_fn({"batch_size": batch_size})
    iterator = predict_data.make_one_shot_iterator().get_next()
    all_prob = []
    all_pre_labels=[]
    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for _ in range(batch_num):
            examples = sess.run(iterator)
            prob = \
                sess.run(probabilities,
                         feed_dict={input_ids: examples["input_ids"],
                                    input_mask: examples["input_mask"],
                                    segment_ids: examples["segment_ids"],
                                    label_ids: examples["label_ids"]})
            all_prob.extend(prob[:, 1].tolist())
            pre_labels = np.argmax(prob, axis=-1).tolist()
            all_pre_labels.extend(pre_labels)
    with open(data_name) as f:
        lines = f.readlines()
    with open(out_file, "w") as f:
        index = 0
        for line in lines:
            if line[0] == ">":
                f.write(line)
            else:
                f.write(line.strip() + " " + str(all_prob[index])+" "+str(all_pre_labels[index]) + "\n")
                index += 1


if __name__ == '__main__':
    main(data_name="./synthensis_peptide.fasta",
         out_file="pre_result/new_model/DPPIV_k1r_rr.txt",
         model_path="out/1kmer_all_data_50_r/model.ckpt",
         kmer=1,
         config_file="./bert_config_1.json",
         vocab_file="./vocab/vocab_1kmer.txt")
