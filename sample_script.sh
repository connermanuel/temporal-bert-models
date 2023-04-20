# Our other baseline is zero-shot pretrained BERT -- no need for further training
python train.py -m bert --data_dir data/tempo_dataset # Finetuned baseline
python train.py -m bert --data_dir data/tempo_dataset_time_token --use_time_token --output_dir outputs/bert_time_token --batch_size 8 # Prepends a time token, finetunes vanilla bert
python train.py -m tempo_bert --data_dir data/tempo_dataset # Finetunes tempobert
python train.py -m orthogonal --data_dir data/tempo_dataset # Finetunes our implementation
