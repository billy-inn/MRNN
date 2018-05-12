from model import Classifier
from utils import embedding_utils
import config
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit

def transform_label(x):
    tmp = np.zeros(4)
    for i in range(x):
        tmp[i] = 1.0
    return tmp

def main():
    df = pd.read_csv(config.ALL_DATA, sep="\t", names=["text", "label"])
    text = list(df.text)
    labels = np.array(list(df.label.map(transform_label))).astype(np.float32)
    embedding = embedding_utils.Embedding(
        config.EMBEDDING_DATA,
        text,
        config.MAX_DOCUMENT_LENGTH
    )

    words = np.array(list(map(embedding._text_transform, text)))
    ss = ShuffleSplit(n_splits=1, test_size=0.1, random_state=config.RANDOM_SEED)
    for train_idx, test_idx in ss.split(np.zeros(len(labels)), labels):
        words_train, words_test = words[train_idx], words[test_idx]
        labels_train, labels_test = labels[train_idx], labels[test_idx]

    kwargs = {
        "seq_len": config.MAX_DOCUMENT_LENGTH,
        "num_classes": 4,
        "pretrained_embedding": embedding.embedding,
        "hparams": {
            "batch_size": 32,
            "num_epochs": 20,
            "lr": 0.001,
            "l2_reg_lambda": 0.001,
        }
    }
    m = Classifier(**kwargs)

    train_set = list(zip(words_train, labels_train))
    test_set = list(zip(words_test, labels_test))

    m.init()
    m.fit(train_set, test_set)

if __name__ == "__main__":
    main()
