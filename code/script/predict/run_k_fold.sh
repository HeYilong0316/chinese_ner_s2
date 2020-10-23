export DATA_DIR=../user_data/k_fold

# lstm+crf
# export MODEL_DIR=../user_data/output_lstm_crf
# export RUN_FILE=script/predict/run_lstm_crf.sh
# sh $RUN_FILE $MODEL_DIR/0_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/1_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/2_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/3_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/4_fold/ $DATA_DIR

# sh $RUN_FILE $MODEL_DIR/5_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/6_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/7_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/8_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/9_fold/ $DATA_DIR

# # idcnn+crf
# export MODEL_DIR=../user_data/output_idcnn_crf
# export RUN_FILE=script/predict/run_idcnn_crf.sh
# sh $RUN_FILE $MODEL_DIR/0_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/1_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/2_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/3_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/4_fold/ $DATA_DIR

# sh $RUN_FILE $MODEL_DIR/5_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/6_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/7_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/8_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/9_fold/ $DATA_DIR

# layer+lstm+crf
export MODEL_DIR=../user_data/output/output_layer_lstm_crf
export RUN_FILE=script/predict/run_layer_lstm_crf.sh
sh $RUN_FILE $MODEL_DIR/0_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/1_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/2_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/3_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/4_fold/ $DATA_DIR

sh $RUN_FILE $MODEL_DIR/5_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/6_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/7_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/8_fold/ $DATA_DIR
sh $RUN_FILE $MODEL_DIR/9_fold/ $DATA_DIR

# layer+idcnn+crf
# export MODEL_DIR=../user_data/output_layer_idcnn_crf
# export RUN_FILE=script/predict/run_layer_idcnn_crf.sh

# # for i in {0..9}:
# # do
# #     sh $RUN_FILE $MODEL_DIR/$i_fold/ $DATA_DIR
# # done

# sh $RUN_FILE $MODEL_DIR/0_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/1_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/2_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/3_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/4_fold/ $DATA_DIR

# sh $RUN_FILE $MODEL_DIR/5_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/6_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/7_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/8_fold/ $DATA_DIR
# sh $RUN_FILE $MODEL_DIR/9_fold/ $DATA_DIR
