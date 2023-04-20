# Temporal Bert Models

Instructions (for training tempo-bert on wikipedia corpus):
* Install pytorch, transformers, and datasets 
* Download the `wiki_dataset` folder and insert it inside `./data`
* Download the `temporal_bert_mlm_base` folder and insert it inside `./models`
* Run `python train_tempo_wikipedia.py -h` to view all CLI arguments. (You should be able to run it without any CLI arguments though.)

Todos:
* Migrate to src layout
* Fix token alignment for mask tokens (mask -> timestamp 0)
* insert special timestamps for time tokens (time -> timestamp 1)
* insert special tokens for times