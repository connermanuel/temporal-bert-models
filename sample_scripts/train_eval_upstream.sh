UPSTREAM_DATA_DIR=data/reddit/processed/news_small
OUTPUT_DIR=output/reddit
RESULTS_DIR=results/reddit

MODEL=bert
ADD_TIME_TOKEN=none
DIR_NAME=bert_1m

python run_train.py	-m $MODEL \
    --task mlm \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/$DIR_NAME" \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16 \
    --auto-batch

python run_evaluate.py	-m $MODEL \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/$DIR_NAME" \
    --results-dir "$RESULTS_DIR/$DIR_NAME" \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16

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