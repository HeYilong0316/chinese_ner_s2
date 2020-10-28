# base
export DATA_DIR=../user_data/data/k_fold_10_510
export OUTPUT_DIR_BASE=../user_data/output

# layer+lstm+crf
export OUTPUT_DIR=$OUTPUT_DIR_BASE/output_layer_lstm_crf
export RUN_TRAIN_FILE=script/train/run_layer_lstm_crf.sh
export RUN_PREDICT_FILE=script/predict/run_layer_lstm_crf.sh

for i in `seq 0 1`
do
    sh $RUN_TRAIN_FILE $DATA_DIR/"fold_"$i 10 2 -1 $OUTPUT_DIR/$i"_fold"
    sh $RUN_PREDICT_FILE $OUTPUT_DIR/$i"_fold"/ $DATA_DIR
    rm $OUTPUT_DIR/$i"_fold"/pytorch_model.bin
done

# layer+idcnn+crf
export OUTPUT_DIR=$OUTPUT_DIR_BASE/output_layer_idcnn_crf
export RUN_TRAIN_FILE=script/train/run_layer_idcnn_crf.sh
export RUN_PREDICT_FILE=script/predict/run_layer_idcnn_crf.sh

for i in `seq 0 1`
do
    sh $RUN_TRAIN_FILE $DATA_DIR/"fold_"$i 10 2 -1 $OUTPUT_DIR/$i"_fold"
    sh $RUN_PREDICT_FILE $OUTPUT_DIR/$i"_fold"/ $DATA_DIR
    rm $OUTPUT_DIR/$i"_fold"/pytorch_model.bin
done











# sh $RUN_FILE $DATA_DIR/fold_0 10 1 -1 $OUTPUT_DIR/0_fold
# sh $RUN_FILE $DATA_DIR/fold_1 10 10 -1 $OUTPUT_DIR/1_fold
# sh $RUN_FILE $DATA_DIR/fold_2 10 10 -1 $OUTPUT_DIR/2_fold
# sh $RUN_FILE $DATA_DIR/fold_3 10 10 -1 $OUTPUT_DIR/3_fold
# sh $RUN_FILE $DATA_DIR/fold_4 10 10 -1 $OUTPUT_DIR/4_fold
# sh $RUN_FILE $DATA_DIR/fold_5 10 10 -1 $OUTPUT_DIR/5_fold
# sh $RUN_FILE $DATA_DIR/fold_6 10 10 -1 $OUTPUT_DIR/6_fold
# sh $RUN_FILE $DATA_DIR/fold_7 10 10 -1 $OUTPUT_DIR/7_fold
# sh $RUN_FILE $DATA_DIR/fold_8 10 10 -1 $OUTPUT_DIR/8_fold
# sh $RUN_FILE $DATA_DIR/fold_9 10 10 -1 $OUTPUT_DIR/9_fold



# sh $RUN_FILE $DATA_DIR/fold_0 10 2 -1 $OUTPUT_DIR/0_fold
# sh $RUN_FILE $DATA_DIR/fold_1 10 1 -1 $OUTPUT_DIR/1_fold
# sh $RUN_FILE $DATA_DIR/fold_2 10 10 -1 $OUTPUT_DIR/2_fold
# sh $RUN_FILE $DATA_DIR/fold_3 10 10 -1 $OUTPUT_DIR/3_fold
# sh $RUN_FILE $DATA_DIR/fold_4 10 10 -1 $OUTPUT_DIR/4_fold
# sh $RUN_FILE $DATA_DIR/fold_5 10 10 -1 $OUTPUT_DIR/5_fold
# sh $RUN_FILE $DATA_DIR/fold_6 10 10 -1 $OUTPUT_DIR/6_fold
# sh $RUN_FILE $DATA_DIR/fold_7 10 10 -1 $OUTPUT_DIR/7_fold
# sh $RUN_FILE $DATA_DIR/fold_8 10 10 -1 $OUTPUT_DIR/8_fold
# sh $RUN_FILE $DATA_DIR/fold_9 10 10 -1 $OUTPUT_DIR/9_fold

# export MODEL_DIR=../user_data/output/output_layer_idcnn_crf
# export RUN_FILE=script/predict/run_layer_idcnn_crf.sh

# for i in `seq 0 1`
# do
#     sh $RUN_FILE $MODEL_DIR/$i"_fold"/ $DATA_DIR
#     rm $MODEL_DIR/$i"_fold"/pytorch_model.bin
# done
