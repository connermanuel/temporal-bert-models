BASE_DIR=/local/scratch/shared/groupdir2/reddit
UPSTREAM_DATA_DIR=$BASE_DIR/data/news_annual_3M
OUTPUT_DIR=$BASE_DIR/output/3M_5e-6
RESULTS_DIR=$BASE_DIR/results/3M_5e-6

ATTENTION=$1
TIME_TOKEN=$2
DIR_NAME=$3

echo "Running with dir_name $3..."
python ../run_train.py \
    -m bert \
    -a $ATTENTION \
    --task mlm \
    --n-contexts 12 \
    --batch-size 256 \
    --num-epochs 5 \
    --lr 5e-6 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/$DIR_NAME" \
    --time-token $TIME_TOKEN \
    --use-fp16 \
    --auto-batch \
    --remove-unused-columns 

python ../run_evaluate.py \
    -m bert \
    -a $ATTENTION \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/$DIR_NAME" \
    --results-dir "$RESULTS_DIR/$DIR_NAME" \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16 \
    --remove-unused-columns 

# The configurations that you will want to run are:

# MODEL=bert
# ADD_TIME_TOKEN=special
# DIR_NAME=token_1m

# MODEL=tempo_bert
# ADD_TIME_TOKEN=none
# DIR_NAME=tempo_bert_1m

# MODEL=orthogonal
# ADD_TIME_TOKEN=none
# DIR_NAME=orthogonal_1m
