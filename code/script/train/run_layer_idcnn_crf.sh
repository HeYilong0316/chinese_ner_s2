export MAX_LENGTH=512
export BERT_MODEL=../user_data/model/chinese-roberta-wwm-large-ext


export DATA_DIR=$1
export TEST_DATA_DIR=$6
export OUTPUT_DIR=$5
export BATCH_SIZE=$2
export NUM_EPOCHS=$3
export WARM_UP=0.15

export SEED=666

export LEARNING_RATE=1e-05
export LW_LEARNING_RATE=1e-04
export LSTM_LEARNING_RATE=1e-03
export IDCNN_LEARNING_RATE=1e-03
export LINEAR_LEARNING_RATE=1e-03
export CRF_LEARNING_RATE=1e-02

export DROPOUT=0.1
export MAX_GRAD_NORM=1.0

export WEIGHT_DECAY=1e-02
export LW_WEIGHT_DECAY=1e-02
export LSTM_WEIGHT_DECAY=1e-02
export IDCNN_WEIGHT_DECAY=1e-02
export LINEAR_WEIGHT_DECAY=1e-02
export CRF_WEIGHT_DECAY=1e-02

export MAX_EPOCH=$4

echo "训练文件路径: " $DATA_DIR
echo "测试文件路径": $TEST_DATA_DIR
echo "BATCH_SIZE:" $BATCH_SIZE
echo "NUM_EPOCHS: " $NUM_EPOCHS
echo "MAX_EPOCH: " $MAX_EPOCH
echo "输出路径: " $OUTPUT_DIR

python -u run_ner.py \
--task_type NER \
--use_crf \
--dropout=$DROPOUT \
--learning_rate $LEARNING_RATE \
--layer_weight_learning_rate $LW_LEARNING_RATE \
--lstm_learning_rate $LSTM_LEARNING_RATE \
--idcnn_learning_rate $IDCNN_LEARNING_RATE \
--crf_learning_rate $CRF_LEARNING_RATE \
--linear_learning_rate $LINEAR_LEARNING_RATE \
--weight_decay $WEIGHT_DECAY \
--layer_weight_weight_decay $LW_WEIGHT_DECAY \
--lstm_weight_decay $LSTM_WEIGHT_DECAY \
--idcnn_weight_decay $IDCNN_WEIGHT_DECAY \
--crf_weight_decay $CRF_WEIGHT_DECAY \
--linear_weight_decay $LINEAR_WEIGHT_DECAY \
--warmup $WARM_UP \
--max_grad_norm $MAX_GRAD_NORM \
--data_dir $DATA_DIR \
--test_data_dir $TEST_DATA_DIR \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--max_epoch $MAX_EPOCH \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--evaluate_during_training \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir \
--use_idcnn \
--multi_layer_fusion \
--gradient_checkpointing \
--disable_tqdm


