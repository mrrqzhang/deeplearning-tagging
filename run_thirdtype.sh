# /bin/bash
set -e

pwd
export BERT_BASE_DIR=/data1/ruiqiang/bert_base_dir/chinese_L-12_H-768_A-12
export DATA_DIR=/data1/ruiqiang/data/thirdtype_mulity
export OUTPUT_DIR=/data1/ruiqiang/model_output/thirdtype_mulity_v2
#CUDA_VISIBLE_DEVICES="0" python2 run_classifier_thirdtype.py \
CUDA_VISIBLE_DEVICES="1" python2 run_classifier_thirdtype_v2.py \
  --task_name=wb \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$DATA_DIR \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config_4layer.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --predict_batch_size=16 \
  --eval_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=$OUTPUT_DIR
