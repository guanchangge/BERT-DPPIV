#Firstly, segment the sequence according to different kmer
#input file:./pre_train_data/uniprot_protein_data.txt 
#output file:./pre_train_data/1kmer_uniprot_protein_data.txt
#kmer size: 1
python pretrain_data_preprocess.py ./pre_train_data/uniprot_protein_data.txt ./pre_train_data/1kmer_uniprot_protein_data.txt 1
# Mask the input token
python create_pretraining_data.py \
	--input_file=./pre_train_data/1kmer_uniprot_protein_data.txt \
	--output_file=./pre_train_data/1kmer_uniprot_protein_data.tfrecord \
	--vocab_file=./vocab/vocab_1kmer.txt \
	--do_lower_case=True \
	--max_seq_length=128 \
	--max_predictions_per_seq=20 \
	--masked_lm_prob=0.15 \
	--random_seed=12345 \
	--dupe_factor=5
