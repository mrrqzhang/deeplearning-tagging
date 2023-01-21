#!/bin/bash



export BERT_BASE_DIR=/data0/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/var_category_top4
export OUTPUT_DIR=../model_output/var_category_4layer_top4
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=1 python run_classifier_varcat.py \
  --task_name=wbmz \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_4layer.json \
  --init_checkpoint=$BERT_BASE_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=32 \
  --predict_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=1.0 \
  --num_variant_tags=4 \
  --output_dir=$OUTPUT_DIR




