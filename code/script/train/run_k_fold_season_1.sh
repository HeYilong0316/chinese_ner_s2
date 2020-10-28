echo "" > train.log

# data size
# 0_fold: 1059
# 1_fold: 1046
# 2_fold: 1058
# 3_fold: 1059
# 4_fold: 1059

# 5_fold: 1054
# 6_fold: 1061
# 7_fold: 1053
# 8_fold: 1056
# 9_fold: 1052

export DATA_DIR=../user_data/data
export OUTPUT_DIR_BASE=$DATA_DIR/output

# batchsize 10
# lstm+crf
export OUTPUT_DIR=$OUTPUT_DIR_BASE/output_lstm_crf
export RUN_FILE=script/train/run_lstm_crf.sh
sh $RUN_FILE $DATA_DIR/k_fold/fold_0 106 159 6 $OUTPUT_DIR/0_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_1 105 156 7 $OUTPUT_DIR/1_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_2 106 159 4 $OUTPUT_DIR/2_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_3 106 159 7 $OUTPUT_DIR/3_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_4 106 159 7 $OUTPUT_DIR/4_fold

sh $RUN_FILE $DATA_DIR/k_fold/fold_5 106 159 6 $OUTPUT_DIR/5_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_6 107 161 9 $OUTPUT_DIR/6_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_7 106 159 7 $OUTPUT_DIR/7_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_8 106 159 5 $OUTPUT_DIR/8_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_9 106 159 7 $OUTPUT_DIR/9_fold


# idcnn+crf
export OUTPUT_DIR=$OUTPUT_DIR_BASE/output_idcnn_crf
export RUN_FILE=script/train/run_idcnn_crf.sh
sh $RUN_FILE $DATA_DIR/k_fold/fold_0 106 159 6 $OUTPUT_DIR/0_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_1 105 156 9 $OUTPUT_DIR/1_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_2 106 159 6 $OUTPUT_DIR/2_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_3 106 159 6 $OUTPUT_DIR/3_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_4 106 159 8 $OUTPUT_DIR/4_fold

sh $RUN_FILE $DATA_DIR/k_fold/fold_5 106 159 9 $OUTPUT_DIR/5_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_6 107 161 7 $OUTPUT_DIR/6_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_7 106 159 5 $OUTPUT_DIR/7_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_8 106 159 5 $OUTPUT_DIR/8_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_9 106 159 10 $OUTPUT_DIR/9_fold


# layer+lstm+crf
export OUTPUT_DIR=$OUTPUT_DIR_BASE/output_layer_lstm_crf
export RUN_FILE=script/train/run_layer_lstm_crf.sh
sh $RUN_FILE $DATA_DIR/k_fold/fold_0 106 159 8 $OUTPUT_DIR/0_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_1 105 156 6 $OUTPUT_DIR/1_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_2 106 159 3 $OUTPUT_DIR/2_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_3 106 159 6 $OUTPUT_DIR/3_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_4 106 159 6 $OUTPUT_DIR/4_fold

sh $RUN_FILE $DATA_DIR/k_fold/fold_5 106 159 6 $OUTPUT_DIR/5_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_6 107 161 4 $OUTPUT_DIR/6_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_7 106 159 5 $OUTPUT_DIR/7_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_8 106 159 5 $OUTPUT_DIR/8_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_9 106 159 6 $OUTPUT_DIR/9_fold


# layer+idcnn+crf
export OUTPUT_DIR=$OUTPUT_DIR_BASE/output_layer_idcnn_crf
export RUN_FILE=script/train/run_layer_idcnn_crf.sh
sh $RUN_FILE $DATA_DIR/k_fold/fold_0 106 159 10 $OUTPUT_DIR/0_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_1 105 156 9 $OUTPUT_DIR/1_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_2 106 159 6 $OUTPUT_DIR/2_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_3 106 159 6 $OUTPUT_DIR/3_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_4 106 159 6 $OUTPUT_DIR/4_fold

sh $RUN_FILE $DATA_DIR/k_fold/fold_5 106 159 10 $OUTPUT_DIR/5_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_6 107 161 7 $OUTPUT_DIR/6_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_7 106 159 5 $OUTPUT_DIR/7_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_8 106 159 7 $OUTPUT_DIR/8_fold
sh $RUN_FILE $DATA_DIR/k_fold/fold_9 106 159 7 $OUTPUT_DIR/9_fold
