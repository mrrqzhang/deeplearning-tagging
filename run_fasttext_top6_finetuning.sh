#!/bin/bash

export BERT_BASE_DIR=../model_output/pretrain_model
#/data0/ruiqiang/tinybert_proj/tf_model_from_pt
#/data0/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/fasttext_top6_output2
export MODEL_OUTPUT_DIR=../model_output/fasttext_top6_recognizer2
CUDA_VISIBLE_DEVICES=1  python run_classifier_ruiqiang.py \
  --task_name=wbml \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_variant_tags=10 \
  --output_dir=$MODEL_OUTPUT_DIR
