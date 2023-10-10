BASE_DIR=/local/scratch/shared/groupdir2/reddit
UPSTREAM_DATA_DIR=$BASE_DIR/data/news_small_trunc
OUTPUT_DIR=$BASE_DIR/output/5e-6
RESULTS_DIR=$BASE_DIR/results/5e-6

MODEL=$1
ADD_TIME_TOKEN=$2
DIR_NAME=$3

python ../run_train.py	-m $MODEL \
    --task mlm \
    --n-contexts 12 \
    --batch-size 256 \
    --num-epochs 5 \
    --lr 5e-6 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/$DIR_NAME" \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16 \
    --auto-batch \

python ../run_evaluate.py -m $MODEL \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/$DIR_NAME" \
    --results-dir "$RESULTS_DIR/$DIR_NAME" \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16 \

# The configurations that you will want to run are:

# MODEL=bert
# ADD_TIME_TOKEN=none
# DIR_NAME=bert_1m

# MODEL=bert
# ADD_TIME_TOKEN=special
# DIR_NAME=token_1m

# MODEL=tempo_bert
# ADD_TIME_TOKEN=none
# DIR_NAME=tempo_bert_1m

# MODEL=orthogonal
# ADD_TIME_TOKEN=none
# DIR_NAME=orthogonal_1m
