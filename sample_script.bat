@REM First: train upstream masked language modeling task on the news dataset

set UPSTREAM_DATA_DIR=data/reddit/processed/news_small_trunc
set OUTPUT_DIR=output_sample/reddit

@REM python run_train.py	-m bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --output-dir "%OUTPUT_DIR%/bert_1M" ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 10

@REM python run_train.py	-m bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --output-dir "%OUTPUT_DIR%/token_1M" ^
@REM     --add-time-token special ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 10

@REM python run_train.py	-m tempo_bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --output-dir "%OUTPUT_DIR%/tempobert_1M" ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 10

@REM python run_train.py	-m orthogonal ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --output-dir "%OUTPUT_DIR%/orthogonal_1M" ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 10

@REM If you want to run evaluation. Note that the auto batch utility doesn't work here.

@REM set RESULTS_DIR=results/sample/reddit
@REM python run_evaluate.py -m bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 64 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/base_1M" ^
@REM     --results-dir "%RESULTS_DIR%/base_1M" ^
@REM     --use-fp16

@REM python run_evaluate.py -m bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 64 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/token_1M" ^
@REM     --results-dir "%RESULTS_DIR%/token_1M" ^
@REM     --add-time-token special ^
@REM     --use-fp16

@REM python run_evaluate.py	-m tempo_bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 64 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/tempobert_1M" ^
@REM     --results-dir "%RESULTS_DIR%/tempobert_1M" ^
@REM     --use-fp16 ^
@REM     --sample 10

@REM python run_evaluate.py	-m orthogonal ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 64 ^
@REM     --data-dir %UPSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/orthogonal_1M" ^
@REM     --results-dir "%RESULTS_DIR%/orthogonal_1M" ^
@REM     --use-fp16 ^
@REM     --sample 10

@REM Then: load the best checkpoint from each architecture and train on the downstream task.
@REM The exact checkpoint number will depend on the batch size and number of epochs, so just be sure to check the output directory when done. 

set DOWNSTREAM_DATA_DIR=data/reddit/processed/psp_small_trunc

set BERT_CHECKPOINT=%OUTPUT_DIR%/bert_1M/checkpoint-5
set TOKEN_CHECKPOINT=%OUTPUT_DIR%/token_1M/checkpoint-5
set TEMPOBERT_CHECKPOINT=%OUTPUT_DIR%/tempobert_1M/checkpoint-5
set ORTHOGONAL_CHECKPOINT=%OUTPUT_DIR%/orthogonal_1M/checkpoint-5

@REM python run_train.py -m bert ^
@REM     --task cls ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %DOWNSTREAM_DATA_DIR% ^
@REM     --pretrain-dir %BERT_CHECKPOINT% ^
@REM     --output-dir "%OUTPUT_DIR%/bert_1M_cls" ^
@REM     --num-labels 5 ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 64

@REM python run_train.py -m bert ^
@REM     --task cls ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %DOWNSTREAM_DATA_DIR% ^
@REM     --pretrain-dir %TOKEN_CHECKPOINT% ^
@REM     --output-dir "%OUTPUT_DIR%/token_1M_cls" ^
@REM     --num-labels 5 ^
@REM     --add-time-token special ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 64

python run_train.py -m tempo_bert ^
    --task cls ^
    --n-contexts 12 ^
    --batch-size 32 ^
    --num-epochs 5 ^
    --lr 5e-5 ^
    --data-dir %DOWNSTREAM_DATA_DIR% ^
    --pretrain-dir %TEMPOBERT_CHECKPOINT% ^
    --output-dir "%OUTPUT_DIR%/tempobert_1M_cls" ^
    --num-labels 5 ^
    --use-fp16 ^
    --auto-batch ^
    --sample 64

@REM python run_train.py	-m orthogonal ^
@REM     --task cls ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --num-epochs 5 ^
@REM     --lr 5e-5 ^
@REM     --data-dir %DOWNSTREAM_DATA_DIR% ^
@REM     --pretrain-dir %ORTHOGONAL_CHECKPOINT% ^
@REM     --output-dir "%OUTPUT_DIR%/orthogonal_1M_cls" ^
@REM     --num-labels 5 ^
@REM     --use-fp16 ^
@REM     --auto-batch ^
@REM     --sample 64

@REM If you want to run evaluation. Note that the auto batch utility doesn't work here

set RESULTS_DIR = results/reddit
@REM python run_evaluate.py -m bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --data-dir %DOWNSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/bert_1M_cls" ^
@REM     --results-dir "%RESULTS_DIR%/bert_1M_cls" ^
@REM     --use-fp16 ^
@REM     --sample 64

@REM python run_evaluate.py -m bert ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --data-dir %DOWNSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/token_1M_cls" ^
@REM     --results-dir "%RESULTS_DIR%/token_1M_cls" ^
@REM     --add-time-token special ^
@REM     --use-fp16 ^
@REM     --sample 64

python run_evaluate.py -m tempo_bert ^
    --task mlm ^
    --n-contexts 12 ^
    --batch-size 32 ^
    --data-dir %DOWNSTREAM_DATA_DIR% ^
    --checkpoint-group-dir "%OUTPUT_DIR%/tempobert_1M_cls" ^
    --results-dir "%RESULTS_DIR%/tempobert_1M_cls" ^
    --use-fp16 ^
    --sample 64

@REM python run_evaluate.py -m orthogonal ^
@REM     --task mlm ^
@REM     --n-contexts 12 ^
@REM     --batch-size 32 ^
@REM     --data-dir %DOWNSTREAM_DATA_DIR% ^
@REM     --checkpoint-group-dir "%OUTPUT_DIR%/orthogonal_1M_cls" ^
@REM     --results-dir "%RESULTS_DIR%/orthogonal_1M_cls" ^
@REM     --use-fp16 ^
@REM     --sample 64