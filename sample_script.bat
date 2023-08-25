@REM First: train upstream masked language modeling task on the news dataset

set UPSTREAM_DATA_DIR="data/reddit/processed/news_small"
set OUTPUT_DIR="output/reddit"

python run_train.py	-m bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 8 ^
    --num-epochs 5 ^
    --lr 5e-5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --output-dir "%OUTPUT_DIR%/bert_1M" ^
    --use-fp16 ^
    --auto-batch

python run_train.py	-m bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 8 ^
    --num-epochs 5 ^
    --lr 5e-5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --output-dir "%OUTPUT_DIR%/token_1M" ^
    --add-time-token special ^
    --use-fp16 ^
    --auto-batch

python run_train.py	-m tempo_bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 8 ^
    --num-epochs 5 ^
    --lr 5e-5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --output-dir "%OUTPUT_DIR%/tempobert_1M" ^
    --use-fp16 ^
    --auto-batch

python run_train.py	-m orthogonal ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 8 ^
    --num-epochs 5 ^
    --lr 5e-5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --output-dir "%OUTPUT_DIR%/orthogonal_1M" ^
    --use-fp16 ^
    --auto-batch

@REM If you want to run evaluation. Note that the auto batch utility doesn't work here.

set RESULTS_DIR = "results/reddit"
python run_evaluate.py	-m bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 4 ^
    --num-epochs 5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --checkpoint-group-dir "%OUTPUT_DIR%/bert_1M" ^
    --results-dir "%RESULTS_DIR%/bert_1M" ^
    --use-fp16

python run_evaluate.py	-m bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 4 ^
    --num-epochs 5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --checkpoint-group-dir "%OUTPUT_DIR%/token_1M" ^
    --results-dir "%RESULTS_DIR%/token_1M" ^
    --add-time-token special ^
    --use-fp16

python run_evaluate.py	-m tempo_bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 4 ^
    --num-epochs 5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --checkpoint-group-dir "%OUTPUT_DIR%/tempobert_1M" ^
    --results-dir "%RESULTS_DIR%/tempobert_1M" ^
    --use-fp16

python run_evaluate.py	-m orthogonal ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 4 ^
    --num-epochs 5 ^
    --data-dir %UPSTREAM_DATA_DIR% ^
    --checkpoint-group-dir "%OUTPUT_DIR%/orthogonal_1M" ^
    --results-dir "%RESULTS_DIR%/orthogonal_1M" ^
    --use-fp16

@REM Then: load the best checkpoint from each architecture and train on the downstream task.
@REM The exact checkpoint number will depend on the batch size and number of epochs, so just be sure to check the output directory when done. 