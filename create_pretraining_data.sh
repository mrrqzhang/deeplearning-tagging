source=./
BERT_BASE_DIR=/data1/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
INPUT=../data/thirdtag_pretrain/train_pretrain_pos.tsv
#INPUT=$source/pretrain/weibo_content_20200201-20200202.5m
OUTPUT=../data/thirdtag_pretrain/weibo_period_pos.tfrecord
#OUTPUT=./pretrain_tfrecord_5m/weibo.tfrecord

CUDA_VISIBLE_DEVICES=1 python $source/create_pretraining_data_wb.py \
  --input_file=$INPUT \
  --output_file=$OUTPUT \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5

