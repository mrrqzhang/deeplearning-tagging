#!/bin/bash



export BERT_BASE_DIR=../roberta_base_dir
export DATA_DIR=../data/huati_material_basis_0227_0406
export OUTPUT_DIR=../model_output/huati_material_basis_0227_0406_roberta
export INIT_CKPT_DIR=$BERT_BASE_DIR
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=1 python run_classifier_py3.py \
  --task_name=wbbc \
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
  --num_variant_tags=1 \
  --output_dir=$OUTPUT_DIR




