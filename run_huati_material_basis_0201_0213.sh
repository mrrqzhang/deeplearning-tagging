#!/bin/bash



export BERT_BASE_DIR=../bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=../data/test_huati/ #pengfei_judged_reorder_biclass.txt    #data/huati_material_basis
export OUTPUT_DIR=../model_output/huati_material_basis_0201_0213
export INIT_CKPT_DIR=$BERT_BASE_DIR
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --task_name=wbbc \
  --do_train=false \
  --do_eval=false \
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




