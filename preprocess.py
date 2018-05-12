import pandas as pd
import config
import re
from nltk.stem.wordnet import WordNetLemmatizer

wn = WordNetLemmatizer()

def clean_text(text, stem_words=True, remove_punctuations=False):
    text = text.lower()
    if remove_punctuations:
        filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        text = text.translate(str.maketrans(filters, ' ' * len(filters)))
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    words = text.split()
    if stem_words:
        words = [wn.lemmatize(w) for w in words]
    if len(words) == 0:
        print("!!!")
    return " ".join(words)

def main():
    df = pd.read_csv(config.ALL_DATA, sep="\t", names=["text", "label"])
    df["len"] = df.text.map(lambda x: len(x.split()))
    print(df.len.max())

if __name__ == "__main__":
    main()
