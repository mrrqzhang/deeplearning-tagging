source=.
BERT_BASE_DIR=/data1/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
#TFRECORD_DIR=./pretrain_tfrecord_5m/weibo.tfrecord

TFRECORD_DIR=../data/thirdtag_pretrain/weibo.tfrecord
DATA_DIR=../data/thirdtag_pretrain/test_inference.tsv
PRETRAIN_MODEL_DIR=../model_output/pretrain_model_thirdtag2


CUDA_VISIBLE_DEVICES=2 python $source/thirdtag_inference.py \
  --input_file=$DATA_DIR \
  --output_dir=$PRETRAIN_MODEL_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_train=False \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=7 \
  --num_train_steps=500000 \
  --num_warmup_steps=1000 \
  --learning_rate=2e-5

