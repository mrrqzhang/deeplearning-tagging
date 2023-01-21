#!/bin/bash

#export BERT_BASE_DIR=/data0/ruiqiang/tinybert_proj/tf_model_from_pt_tfformat
export BERT_BASE_DIR=../model_output/pretrain_model_weibo_pretrain_8_days   #/data0/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/fasttext_multione_training
export OUTPUT_DIR=../model_output/fasttext_multione_training_4layer_weibo_pretrain
#export ckpt=converted_model-tf_model.ckpt
export ckpt=model.ckpt

CUDA_VISIBLE_DEVICES=2  python run_classifier.py \
  --task_name=mrpc \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_4layer.json \
  --init_checkpoint=$BERT_BASE_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=8 \
  --predict_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=6.0 \
  --num_variant_tags=10 \
  --output_dir=$OUTPUT_DIR
