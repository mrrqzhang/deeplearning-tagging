#!/bin/bash



export BERT_BASE_DIR=../bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/thirdtag_exinfos
export OUTPUT_DIR=../model_output/thirdtag_varcat
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=2 python run_classifier_varcat_deepwide_py3.py \
  --task_name=wbmz \
  --do_train=false \
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_4layer.json \
  --init_checkpoint=$BERT_BASE_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=1 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_variant_tags=4 \
  --num_exinfos_hidden_layers=0 \
  --exinfos_length=0 \
  --output_dir=$OUTPUT_DIR




