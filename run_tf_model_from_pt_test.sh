#!/bin/bash

export BERT_BASE_DIR=../model_output/small_sample   #/data0/ruiqiang/tinybert_proj/tf_model_from_pt_tfformat
#/data0/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/small_sample
export OUTPUT_DIR=../model_output/small_sample
CUDA_VISIBLE_DEVICES=3  python run_classifier.py \
  --task_name=mrpc \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/model.ckpt-374 \
  --max_seq_length=128 \
  --train_batch_size=8 \
  --predict_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_variant_tags=10 \
  --output_dir=$OUTPUT_DIR
