# BERT-DPPIV
A novel attention-based peptide language model for DPP-IV inhibitory peptide identification
This is a model for DPP-IV inhibitory peptides recognition and prediction based on BERT which is proposed by Google AI. We pre-train a BERT model through amount of proteins sequences downloaded from UniPort and fine-tune the model on DPP-IV dataset and evaluate its performance.
If you want to pre-train the BERT model, you should download the pre-train data from https://drive.google.com/file/d/1QeXWV5_OIKgms7u5ShNfvPEKPBLOC3UT/view?usp=sharing. Then you should use the script pretrain_data_creating.sh to preprocess the pre-train data.

chmod 764 ./pretrain_data_creating.sh

./pretrain_data_creating.sh

After preprocessing the data, then you can use the script pre_train.sh to pre-train the BERT model.
chmod 764 ./pre_train.sh
./pre_train.sh

