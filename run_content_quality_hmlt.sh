#!/bin/bash



export BERT_BASE_DIR=/data/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/content_quality_hmlt
export OUTPUT_DIR=../model_output/content_quality_hmlt
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=1 python run_classifier_varcat_content.py \
  --task_name=wbcq \
  --do_train=false \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_variant_tags=1 \
  --output_dir=$OUTPUT_DIR




