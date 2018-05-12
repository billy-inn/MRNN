import os
import config
import csv
import re
from nltk.stem.wordnet import WordNetLemmatizer

wn = WordNetLemmatizer()

lvl1 = ['"medium"']
lvl2 = ['"high"', '"extreme"']
pos = ['"positive"', '"uncertain-positive"']
neg = ['"negative"', '"uncertain-negative"']
neu = ['"neutral"', '"uncertain-neutral"']

def clean_text(text, stem_words=True, remove_punctuations=True):
    text = text.lower()

    text = re.sub(r"\n", " ", text)         # '\n'      --> ' '
    text = re.sub(r"\'s", " \'s", text)      # it's      --> it 's
    text = re.sub(r"\'ve", " have", text)   # they've   --> they have
    text = re.sub(r"\'t", " not", text)    # can't     --> can not
    text = re.sub(r"\'re", " are", text)    # they're   --> they are
    text = re.sub(r"\'d", "", text)         # I'd (I had, I would) --> I
    text = re.sub(r"\'ll", " will", text)   # I'll      --> I will
    text = re.sub(r" i m ", " i am ", text)
    text = re.sub(r" i'm ", " i am ", text)
    text = re.sub(r" can't ", " cannot ", text)

    # punctuation
    text = re.sub(r"\"", " \" ", text)       # "a"       --> " a "
    text = re.sub(r"\.", " . ", text)       # they.     --> they .
    text = re.sub(r"\,", " , ", text)       # they,     --> they ,
    text = re.sub(r"\-", " ", text)         # "low-cost"--> lost cost
    text = re.sub(r"\–", " ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\!", " ! ", text)
    text = re.sub(r"\]", " ] ", text)
    text = re.sub(r"\[", " [ ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"\<", " < ", text)
    text = re.sub(r"\>", " > ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\;", " ; ", text)
    text = re.sub(r"\:", " : ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"\$", " $ ", text)
    text = re.sub(r"\_", " _ ", text)
    text = re.sub(r"\\", " ", text)
    text = re.sub(r"/", " ", text)

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


def tokenize(s, delimiter):
    tokens = []
    j = 0
    s += delimiter
    quoted = False
    for i in range(len(s)):
        if s[i] == '"':
            quoted = not quoted
        if s[i] == delimiter and not quoted:
            tokens.append(s[j:i])
            j = i + 1
    return tokens

class Doc:
    def __init__(self, data_home, parent, leaf, ver):
        self.ver = ver
        f = open(os.path.join(data_home, "docs", parent, leaf))
        self.text = f.read()
        f.close()

        self.annotations = []
        f = open(os.path.join(data_home, "man_anns", parent, leaf,
                              "gateman.mpqa.lre.%s" % ver))
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row[0].strip()[0] == "#":
                continue
            l, r = row[1].split(",")
            l, r = int(l), int(r)
            if row[3] != 'GATE_expressive-subjectivity':
                continue
            ann = dict(left=l, right=r)
            if len(row) > 4:
                kv = tokenize(row[4].strip(), " ")
                meta = dict([t.split('=') for t in kv if len(t) > 0])
                ann.update(meta)
            self.annotations.append(ann)
        f.close()

    def sentiment_phrases(self):
        for ann in self.annotations:
            if 'intensity' not in ann:
                continue
            if 'polarity' not in ann:
                continue
            phrase = self.text[ann['left']:ann['right']]
            if ann['polarity'] in neg:
                if ann['intensity'] in lvl2:
                    yield phrase, 0
                elif ann['intensity'] in lvl1:
                    yield phrase, 1
            elif ann['polarity'] in neu:
                if ann['intensity'] in (lvl1 + lvl2):
                    yield phrase, 2
            elif ann['polarity'] in pos:
                if ann['intensity'] in lvl1:
                    yield phrase, 3
                elif ann['intensity'] in lvl2:
                    yield phrase, 4

def main():
    path = config.MPQA_PATH
    ver = path[-3:]
    docs_path = os.path.join(path, "docs")
    docs = []
    for parent in os.listdir(docs_path):
        for leaf in os.listdir(os.path.join(docs_path, parent)):
            docs.append(Doc(path, parent, leaf, ver))
    writer = csv.writer(open(config.ALL_DATA, "w"), delimiter="\t")
    for doc in docs:
        for phrase, label in doc.sentiment_phrases():
            writer.writerow([clean_text(phrase), label])

if __name__ == "__main__":
    main()
