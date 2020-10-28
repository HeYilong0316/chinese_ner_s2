echo "model path: $1"
echo "data path: $2"

export MAX_LENGTH=512
export BERT_MODEL=$1
export DATA_DIR=$2
export OUTPUT_DIR=$BERT_MODEL
export BATCH_SIZE=32

python -u run_ner.py \
--task_type NER \
--use_crf \
--per_device_eval_batch_size $BATCH_SIZE \
--data_dir $DATA_DIR \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--do_predict \
--use_idcnn \
--multi_layer_fusion