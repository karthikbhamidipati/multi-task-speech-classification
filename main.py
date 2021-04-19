from argparse import ArgumentParser

from models.train import train_model
from utils.preprocess import preprocess, extract_features


def is_sub_arg(arg):
    key, value = arg
    return value is not None and key != 'action'


def clean_args(args):
    action = args.action
    cleaned_args = dict(filter(is_sub_arg, args._get_kwargs()))
    return action, cleaned_args


def main():
    parser = ArgumentParser()
    action_parser = parser.add_subparsers(title="actions", dest="action", required=True,
                                          help="select action to execute")

    # args for preprocessing
    preprocess_parser = action_parser.add_parser("preprocess", help="preprocess data")
    preprocess_parser.add_argument("-r", "--root-dir", dest="root_dir", required=True,
                                   help="root directory of the common voice dataset")

    # args for feature extraction
    feature_extractor_parser = action_parser.add_parser("feature_extractor", help="feature extractor")
    feature_extractor_parser.add_argument("-r", "--root-dir", dest="root_dir", required=True,
                                          help="root directory of the common voice dataset")

    # args for training
    training_parser = action_parser.add_parser("train", help="Train the model")
    training_parser.add_argument("-r", "--root-dir", dest="root_dir", required=True,
                                 help="root directory of the common voice dataset")
    training_parser.add_argument("-m", "--model-name", dest="model_key", required=True,
                                 help="root directory of the common voice dataset")

    action, args = clean_args(parser.parse_args())

    if action == 'preprocess':
        preprocess(**args)
    elif action == 'feature_extractor':
        extract_features(**args)
    elif action == 'train':
        train_model(**args)


if __name__ == '__main__':
    main()
