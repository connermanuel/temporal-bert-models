BASE_DIR=/local/scratch/shared/groupdir2/reddit
OUTPUT_DIR=$BASE_DIR/output/5e-6 # The same output dir as the upstream task.
DOWNSTREAM_DATA_DIR=$BASE_DIR/data/psp_small_trunc
RESULTS_DIR=$BASE_DIR/results/5e-6

MODEL=$1
ADD_TIME_TOKEN=$2
MLM_DIR_NAME=$3 # The specific folder where you stored the checkpoints for this model.
CLS_DIR_NAME=$3_cls

CHECKPOINT_DIR=$OUTPUT_DIR/$MLM_DIR_NAME/checkpoint-19425 # Change this number!

#python ../run_train.py	-m $MODEL \
#    --task cls \
#    --n-contexts 12 \
#    --batch-size 256 \
#    --num-epochs 5 \
#    --lr 5e-6 \
#    --data-dir $DOWNSTREAM_DATA_DIR \
#    --pretrain-dir $CHECKPOINT_DIR \
#    --output-dir "$OUTPUT_DIR/$CLS_DIR_NAME" \
#    --num-labels 5 \
#    --add-time-token $ADD_TIME_TOKEN \
#    --use-fp16 \
#    --auto-batch

python ../run_evaluate.py -m $MODEL \
    --task cls \
    --n-contexts 12 \
    --batch-size 256 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/$CLS_DIR_NAME" \
    --results-dir "$RESULTS_DIR/$CLS_DIR_NAME" \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16

# The configurations that you will want to run are:

# MODEL=bert
# ADD_TIME_TOKEN=none
# MLM_DIR_NAME=bert_1m
# CLS_DIR_NAME=bert_1m_cls

# MODEL=bert
# ADD_TIME_TOKEN=special
# MLM_DIR_NAME=token_1m
# CLS_DIR_NAME=token_1m_cls

# MODEL=tempo_bert
# ADD_TIME_TOKEN=none
# MLM_DIR_NAME=tempo_bert_1m
# CLS_DIR_NAME=tempo_bert_1m_cls

# MODEL=orthogonal
# ADD_TIME_TOKEN=none
# MLM_DIR_NAME=orthogonal_1m
# CLS_DIR_NAME=orthogonal_1m_cls
