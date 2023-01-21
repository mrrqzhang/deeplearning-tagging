source=.
BERT_BASE_DIR=/data0/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
#TFRECORD_DIR=./pretrain_tfrecord_5m/weibo.tfrecord

TFRECORD_DIR=../data/weibo_pretrain_8_days/weibo.tfrecord

PRETRAIN_MODEL_DIR=../model_output/pretrain_model_weibo_pretrain_8_days


CUDA_VISIBLE_DEVICES=1 python $source/run_pretraining.py \
  --input_file=$TFRECORD_DIR \
  --output_dir=$PRETRAIN_MODEL_DIR \
  --do_train=True \
  --do_eval=False \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=2000000 \
  --num_warmup_steps=1000 \
  --learning_rate=2e-5

