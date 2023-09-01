# First: train upstream masked language modeling task on the news dataset

UPSTREAM_DATA_DIR=data/reddit/processed/news_small
OUTPUT_DIR=output/reddit

python run_train.py	-m bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/bert_1M" \
    --use-fp16 \
    --auto-batch

python run_train.py	-m bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/token_1M" \
    --add-time-token special \
    --use-fp16 \
    --auto-batch

python run_train.py	-m tempo_bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/tempobert_1M" \
    --use-fp16 \
    --auto-batch

python run_train.py	-m orthogonal \
    --task mlm \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --output-dir "$OUTPUT_DIR/orthogonal_1M" \
    --use-fp16 \
    --auto-batch

# If you want to run evaluation. Note that the auto batch utility doesn't work here.

RESULTS_DIR = results/reddit
python run_evaluate.py	-m bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/bert_1M" \
    --results-dir "$RESULTS_DIR/bert_1M" \
    --use-fp16

python run_evaluate.py	-m bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/token_1M" \
    --results-dir "$RESULTS_DIR/token_1M" \
    --add-time-token special \
    --use-fp16

python run_evaluate.py	-m tempo_bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/tempobert_1M" \
    --results-dir "$RESULTS_DIR/tempobert_1M" \
    --use-fp16

python run_evaluate.py	-m orthogonal \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $UPSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/orthogonal_1M" \
    --results-dir "$RESULTS_DIR/orthogonal_1M" \
    --use-fp16

# Then: load the best checkpoint from each architecture and train on the downstream task.
# The exact checkpoint number will depend on the batch size and number of epochs, so just be sure to check the output directory when done. 

DOWNSTREAM_DATA_DIR = data/reddit/processed/news_psp

BERT_CHECKPOINT = $OUTPUT_DIR/bert_1M/checkpoint-212500
TOKEN_CHECKPOINT = $OUTPUT_DIR/token_1M/checkpoint-212500
TEMPOBERT_CHECKPOINT = $OUTPUT_DIR/tempobert_1M/checkpoint-212500
ORTHOGONAL_CHECKPOINT = $OUTPUT_DIR/orthogonal_1M/checkpoint-212500

python run_train.py	-m bert \
    --task cls \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --pretrain-dir $BERT_CHECKPOINT \
    --output-dir "$OUTPUT_DIR/bert_1M_cls" \
    --num-labels 5 \
    --use-fp16 \
    --auto-batch

python run_train.py	-m bert \
    --task cls \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --pretrain-dir $TOKEN_CHECKPOINT \
    --output-dir "$OUTPUT_DIR/token_1M_cls" \
    --num-labels 5 \
    --add-time-token special \
    --use-fp16 \
    --auto-batch

python run_train.py	-m tempo_bert \
    --task cls \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --pretrain-dir $TEMPOBERT_CHECKPOINT \
    --output-dir "$OUTPUT_DIR/tempobert_1M_cls" \
    --num-labels 5 \
    --use-fp16 \
    --auto-batch

python run_train.py	-m orthogonal \
    --task cls \
    --n-contexts 12 \
    --batch-size 512 \
    --num-epochs 5 \
    --lr 5e-5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --pretrain-dir $ORTHOGONAL_CHECKPOINT \
    --output-dir "$OUTPUT_DIR/orthogonal_1M_cls" \
    --num-labels 5 \
    --use-fp16 \
    --auto-batch

# If you want to run evaluation. Note that the auto batch utility doesn't work here.

RESULTS_DIR = results/reddit
python run_evaluate.py	-m bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/bert_1M_cls" \
    --results-dir "$RESULTS_DIR/bert_1M_cls" \
    --use-fp16

python run_evaluate.py	-m bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/token_1M_cls" \
    --results-dir "$RESULTS_DIR/token_1M_cls" \
    --add-time-token special \
    --use-fp16

python run_evaluate.py	-m tempo_bert \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/tempobert_1M_cls" \
    --results-dir "$RESULTS_DIR/tempobert_1M_cls" \
    --use-fp16

python run_evaluate.py	-m orthogonal \
    --task mlm \
    --n-contexts 12 \
    --batch-size 64 \
    --num-epochs 5 \
    --data-dir $DOWNSTREAM_DATA_DIR \
    --checkpoint-group-dir "$OUTPUT_DIR/orthogonal_1M_cls" \
    --results-dir "$RESULTS_DIR/orthogonal_1M_cls" \
    --use-fp16