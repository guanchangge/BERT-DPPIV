# BERT-DPPIV
A novel attention-based peptide language model for DPP-IV inhibitory peptide identification
This is a model for DPP-IV inhibitory peptides recognition and prediction based on BERT which is proposed by Google AI. We pre-train a BERT model through amount of proteins sequences downloaded from UniPort and fine-tune the model on DPP-IV dataset and evaluate its performance.


If you want to pre-train the BERT model, you should download the pre-train data from >https://drive.google.com/file/d/1QeXWV5_OIKgms7u5ShNfvPEKPBLOC3UT/view?usp=sharing. 

# Pre-training

You should use the script pretrain_data_creating.sh to preprocess the pre-train data.

```
chmod 764 ./pretrain_data_creating.sh

./pretrain_data_creating.sh
```
After preprocessing the data, then you can use the script pre_train.sh to pre-train the BERT model.
```
chmod 764 ./pre_train.sh

./pre_train.sh
```
# Fine-tune
When you ready to fine-tune the model , you should run the following code. The format of positive file and negative file is fasta file.
```
python fine_tune_model.py \
--do_eval True \
--do_save_model True \
--data_name DPP-IV \
--batch_size 32 \
--num_train_epochs 50 \
--warmup_proportion 0.1 \
--learning_rate 2e-6 \
--using_tpu False \
--seq_length 128 \
--data_root ./Fine_tune_data/1kmer_data/ \
--positive_file train-positive.txt \
--negative_file train-negative.txt \
--kmer 1 \
--vocab_file ./vocab/vocab_1kmer.txt \
--init_checkpoint ./model/1kmer_model/model.ckpt \
--bert_config ./bert_config_1.json \
--save_path ./fine_tune_model/1kmer_fine_tune_model/model.ckpt
```
The meaning of each parameter is as follows, you should change these according to your needs. You can also open file fine_tune_model.py and change the  these parameters.

> do_eval: whether to evaluate the model after training\
> do_save_model: whether to save the model after training\
> data_name: the name of the training set\
> batch_size: batch size\
> num_train_epochs: training epochs\
> warmup_proportion: proportion of warmup\
> learning_rate: learning rate\
> using_tpu: Whether to use TPU\
> seq_length: sequence length\
> data_root: the location of the training set to be used\
> positive_file: The name of file containing the positive trian data\
> negative_file: The name of file containing the negative trian data\
> kmer: The type of word segmentation\
> vocab_file: location of dictionary\
> init_checkpoint: initialization node of the model\
> bert_config: BERT configuration\
> save_path: where to save the trained model\

# Prediction

You can predict your peptides data by command
'''
python BERT_DPPIV_prediction.py
'''
You should change the codes in row 140-145 according to your needs.

> data_name: location of the testing set\
> out_file: storage location of test results\
> model_path: the location of the trained model\
> kmer: the type of word segmentation\
> config_file: BERT configuration\
> vocab_file: location of dictionary\
