"""Common arguments for pipelines using the models (e.g. training/evaluation)"""

import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    model_args = parser.add_argument_group("Model arguments")
    model_args.add_argument(
        "-m",
        "--model_architecture",
        help="The model architecture to train.",
        choices=["bert", "t5"],
        required=True,
    )
    model_args.add_argument(
        "-a",
        "--attention",
        help="The variant of attention to use.",
        choices=["base", "tempo_bert", "orthogonal"],
        required=True,
    )
    model_args.add_argument(
        "--task",
        help="The task to train on.",
        type=str,
        choices=["mlm", "cls", "ssm"],
        required=True,
    )
    model_args.add_argument(
        "--n-contexts",
        help="Number of contexts/timestamps.",
        type=int,
    )
    model_args.add_argument(
        "--pretrain-dir",
        help="Location of pretrained model. Required for classification task.",
        type=str,
        default=None,
    )
    model_args.add_argument(
        "--num-labels",
        help="Number of labels. Required for classification task.",
        type=int,
        default=None,
    )
    model_args.add_argument(
        "--alpha",
        help="Regularization parameter. Defaults to 0. Only used for orthogonal model.",
        type=float,
        default=0,
    )

    data_args = parser.add_argument_group("Data arguments")
    data_args.add_argument("--data-dir", help="Path of the huggingface dataset.")
    data_args.add_argument(
        "--time-token",
        help="Modifies the dataset to insert generic special time tokens. Use 'string' for tokenized strings, and 'special' for inserted special tokens.",
        choices=["none", "string", "special"],
        default="none",
    )
    data_args.add_argument(
        "--sample",
        help="Indicates how many documents to use. If unset, uses the entire dataset.",
        type=int,
        default=0,
    )
    data_args.add_argument(
        "--start-year",
        help="The year to start writing timestamps",
        type=int,
        default=2010,
    )
    data_args.add_argument(
        "--save-dataset",
        help="After processing, stores the dataset to this location.",
        default=None,
    )
    data_args.add_argument(
        "--remove-unused-columns",
        help="Removes any columns not passed into the model's \"forward\" call before inserting into the trainer. Might need this for MLM and CLS, probably will break SSM.",
        action="store_true",
    )
    data_args.add_argument(
        "--skip-process",
        help="Skips the processing of the dataset (shuffling and prepending time tokens.)",
        action="store_true",
    )

    return parser
