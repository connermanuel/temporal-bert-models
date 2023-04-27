# Our other baseline is zero-shot pretrained BERT -- no need for further training
python train.py -m bert --data-dir data/tempo_dataset --output-dir outputs/bert --use-fp16 # Finetuned baseline

python train.py -m bert --data-dir data/tempo_dataset_time_token --use_time_token string --output-dir outputs/bert_time_token --batch-size 8 --use-fp16  # Prepends a time token, finetunes vanilla bert
python train.py -m bert --data-dir data/tempo_dataset_time_special_token --use_time_token special --output-dir outputs/bert_time__special_token --batch-size 8 --use-fp16 # Prepends a special time token, finetunes vanilla bert

python train.py -m tempo_bert --data-dir data/tempo_dataset --output-dir outputs/tempo_bert --use-fp16 # Finetunes tempobert

# Finetunes our implementation with differenta alphas
python train.py -m orthogonal --data-dir data/tempo_dataset --output-dir outputs/orthogonal_1 --use-fp16
python train.py -m orthogonal --data-dir data/tempo_dataset --alpha 10 --output-dir outputs/orthogonal_10 --use-fp16
python train.py -m orthogonal --data-dir data/tempo_dataset --alpha 100 --output-dir outputs/orthogonal_10 --use-fp16
python train.py -m orthogonal --data-dir data/tempo_dataset --alpha 0.1 --output-dir outputs/orthogonal_0.1 --use-fp16
python train.py -m orthogonal --data-dir data/tempo_dataset --alpha 0.01 --output-dir outputs/orthogonal_0.01 --use-fp16
python train.py -m orthogonal --data-dir data/tempo_dataset --alpha 0 --output-dir outputs/orthogonal_0 --use-fp16
