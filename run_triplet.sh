#!/bin/bash



export BERT_BASE_DIR=../bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/thirdtag_triplet
export OUTPUT_DIR=../model_output/thirdtag_triplet  #output embedding
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=2 python3 run_temp.py \
  --task_name=wb_triplet \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
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
  --num_variant_tags=2 \
  --exinfos_length=128 \
  --output_dir=$OUTPUT_DIR




