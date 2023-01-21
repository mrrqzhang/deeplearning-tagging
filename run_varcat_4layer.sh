#!/bin/bash



export BERT_BASE_DIR=/data2/ruiqiang/chinese_simbert_L-12_H-768_A-12
export DATA_DIR=../data/4000w3sm #thirdtag_xiaocao
export OUTPUT_DIR=/data2/ruiqiang/model_output/simcse_open_model
export ckpt=bert_model.ckpt

CUDA_VISIBLE_DEVICES=1 python run_varcat_py3.py \
  --task_name=wbmz \
  --do_train=true\
  --do_eval=false \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_4layer.json \
  --init_checkpoint=$BERT_BASE_DIR/$ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=32 \
  --eval_batch_size=16 \
  --learning_rate=2e-5 \
  --num_train_epochs=100.0 \
  --num_variant_tags=4 \
  --output_dir=$OUTPUT_DIR




