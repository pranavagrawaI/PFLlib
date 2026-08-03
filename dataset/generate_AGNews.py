import numpy as np
import os
import sys
import csv
import json
import argparse
import random
from utils.dataset_utils import check, separate_data, split_data, save_file
from utils.language_utils import basic_english_tokenizer, build_vocab, encode_tokens


random.seed(1)
np.random.seed(1)
num_clients = 20
max_len = 200
max_tokens = 32000
dir_path = "AGNews/"


def _read_agnews_csv(root):
    """Read cached AG_NEWS CSVs (label, title, description) -> (labels, texts).
    Text = title + ' ' + description, matching torchtext's AG_NEWS join.
    Falls back to the shared AGNews/rawdata cache if this out_dir has none."""
    candidates = [
        os.path.join(root, "datasets", "AG_NEWS"),
        os.path.join("AGNews", "rawdata", "datasets", "AG_NEWS"),
    ]
    base = next((c for c in candidates if os.path.exists(os.path.join(c, "train.csv"))), None)
    if base is None:
        raise FileNotFoundError(
            "AG_NEWS train.csv/test.csv not found. Expected under "
            f"{candidates[0]} or the shared AGNews/rawdata cache.")

    def rd(split):
        labels, texts = [], []
        with open(os.path.join(base, f"{split}.csv"), newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                labels.append(int(row[0]))
                texts.append((row[1].strip() + " " + row[2].strip()).strip())
        return labels, texts

    return rd("train"), rd("test")


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition, class_per_client=None, seq_len=max_len):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition, class_per_client):
        return

    # Read cached AG_NEWS CSVs (torchtext-free).
    (trainlabel, traintext), (testlabel, testtext) = _read_agnews_csv(dir_path + "rawdata")

    # Tokenize once; build the vocab from the TRAIN split only (deterministic),
    # then encode train+test with that frozen vocab. Both practical/pathological
    # splits run this identically, so they share the same processed dataset and
    # differ only in partitioning.
    train_tokens = [basic_english_tokenizer(t) for t in traintext]
    test_tokens = [basic_english_tokenizer(t) for t in testtext]
    itos, stoi = build_vocab(train_tokens, max_tokens=max_tokens)

    all_tokens = train_tokens + test_tokens
    dataset_label = list(trainlabel) + list(testlabel)
    num_classes = len(set(dataset_label))
    print(f'Number of classes: {num_classes}')

    encoded = [encode_tokens(toks, stoi, seq_len) for toks in all_tokens]
    text_list = np.array([(ids, length) for ids, length in encoded], dtype=object)
    label_list = np.array([int(l) - 1 for l in dataset_label])

    X, y, statistic = separate_data((text_list, label_list), num_clients, num_classes, niid, balance, partition, class_per_client=class_per_client)
    train_data, test_data = split_data(X, y)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes,
            statistic, niid, balance, partition, class_per_client)

    # Persist vocab + preprocessing metadata so the pipeline is reproducible.
    with open(dir_path + "vocab.json", "w", encoding="utf-8") as f:
        json.dump({"itos": itos}, f)
    with open(dir_path + "preprocess_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "tokenizer": "basic_english (stdlib reimplementation, torchtext-free)",
            "vocab_source": "train split only",
            "vocab_size": len(itos),
            "max_tokens": max_tokens,
            "max_len": seq_len,
            "specials": ["<pad>", "<cls>", "<unk>", "<eos>"],
            "cls_prepended": True,
            "pad_id": 0,
            "label_pipeline": "int(class_index) - 1",
            "num_train": len(traintext),
            "num_test": len(testtext),
            "num_classes": num_classes,
        }, f, indent=2)

    print("The size of vocabulary:", len(itos))


if __name__ == "__main__":
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        niid = True if sys.argv[1] == "noniid" else False
        balance = True if sys.argv[2] == "balance" else False
        partition = sys.argv[3] if sys.argv[3] != "-" else None
        generate_dataset(dir_path, num_clients, niid, balance, partition)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--out_dir", type=str, default=dir_path)
        parser.add_argument("--num_clients", type=int, default=num_clients)
        parser.add_argument("--niid", action="store_true", default=True)
        parser.add_argument("--iid", dest="niid", action="store_false")
        parser.add_argument("--balance", type=lambda v: str(v).lower() in ("1", "true", "yes", "y"), default=True)
        parser.add_argument("--partition", type=str, default="dir", choices=["dir", "pat", "exdir"])
        parser.add_argument("--class_per_client", type=int, default=None)
        parser.add_argument("--max_len", type=int, default=max_len)
        args = parser.parse_args()
        generate_dataset(args.out_dir, args.num_clients, args.niid, args.balance, args.partition, args.class_per_client, seq_len=args.max_len)
