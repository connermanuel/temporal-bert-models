import argparse
from datasets import load_from_disk

def extend_by_one(examples, key, value):
    for i in range(len(examples[key])):
        examples[key][i].append(value)
    return examples

def fix_dataset(dataset_path, destination):
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(lambda x: extend_by_one(x, "attention_mask", 1), batched=True)
    dataset = dataset.map(lambda x: extend_by_one(x, "token_type_ids", 0), batched=True)
    dataset.save_to_disk(destination)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixes the special time token dataset.")
    parser.add_argument('path',                        
        help="The path of the special token dataset")
    parser.add_argument('-d', '--destination',                        
        help="The path to save the fixed dataset to. If unset, will just append '_fixed' to the end of the existing path. Note that you cannot overwrite the existing dataset directly.", default=None)
    args = parser.parse_args()
    if args.destination is None:
        args.destination = args.path + "_fixed"
    fix_dataset(args.path, args.destination)