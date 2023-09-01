OUTPUT_DIR=output/reddit # The same output dir as the upstream task.
DOWNSTREAM_DATA_DIR=data/reddit/processed/news_psp
RESULTS_DIR=results/reddit

MODEL=bert
ADD_TIME_TOKEN=none
MLM_DIR_NAME=bert_1m # The specific folder where you stored the checkpoints for this model.
CLS_DIR_NAME=bert_1m_cls

CHECKPOINT_DIR=$OUTPUT_DIR/$MLM_DIR_NAME/checkpoint-212500 # Change this number!

python run_train.py	-m $MODEL \
    --task cls \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --pretrain-dir $CHECKPOINT_DIR \
    --output-dir "$OUTPUT_DIR/$DIR_NAME" \
    --num-labels 5 \
    --add-time-token $ADD_TIME_TOKEN \
    --use-fp16 \
    --auto-batch

python run_evaluate.py	-m $MODEL \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/$DIR_NAME" \
    --results-dir "$RESULTS_DIR/$DIR_NAME" \
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