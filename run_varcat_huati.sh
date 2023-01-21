#!/bin/bash

#huati_material_basis_0201_0316
init_dir=../model_output/huati_material_basis_0201_0316
init_model=model.ckpt-2316000


export BERT_BASE_DIR=../bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/var_huati_2
export OUTPUT_DIR=../model_output/var_huati_2
export INIT_CKPT_DIR=$init_dir
export ckpt=$init_model


CUDA_VISIBLE_DEVICES=1 python run_classifier_varcat.py \
  --task_name=wbmz \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_4layer.json \
  --init_checkpoint=$INIT_CKPT_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --num_variant_tags=5 \
  --output_dir=$OUTPUT_DIR




