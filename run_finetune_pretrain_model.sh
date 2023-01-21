#!/bin/bash

export BERT_CONF_DIR=../bert_base_dir/chinese_L-12_H-768_A-12
export BERT_INIT_DIR=../model_output/pretrain_model_thirdtag
export DATA_DIR=../data/thirdtag_pretrain
export OUTPUT_DIR=../model_output/thirdtag_finetune_pretrain_model
#export ckpt=converted_model-tf_model.ckpt
export ckpt=model.ckpt-500000
export bert_ckpt=bert_model.ckpt

#batch_size must <16
CUDA_VISIBLE_DEVICES=1  python3 run_classifier_varcat_deepwide_py3.py \
  --task_name=wb_binary_ab \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_CONF_DIR/vocab.txt \
  --bert_config_file=$BERT_CONF_DIR/bert_config_4layer.json \
  --init_checkpoint=$BERT_INIT_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=1 \
  --eval_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_variant_tags=2 \
  --exinfos_length=0 \
  --output_dir=$OUTPUT_DIR




