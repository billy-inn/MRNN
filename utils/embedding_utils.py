import numpy as np
import gensim

class Embedding:
    def __init__(self, f, corpus, max_document_length):
        if ".txt" in f:
            model = gensim.models.KeyedVectors.load_word2vec_format(f, binary=False)
        else:
            model = gensim.models.KeyedVectors.load_word2vec_format(f, binary=True)

        wordSet = set()
        for sen in corpus:
            words = sen.split()
            for w in words:
                if w in model:
                    wordSet.add(w)

        vocab_size = len(wordSet)
        print("%d unique tokens have been found!" % vocab_size)
        embedding_dim = model.syn0.shape[1]
        word2id = {"<PAD>": 0}
        id2word = {0: "<PAD>"}
        word2id = {"<UNK>": 1}
        id2word = {1: "<UNK>"}
        embedding = np.zeros((vocab_size + 2, embedding_dim))

        np.random.seed(0)
        embedding[1, :] = np.random.uniform(-1, 1, embedding_dim)
        for i, word in enumerate(sorted(wordSet)):
            word2id[word] = i + 2
            id2word[i + 2] = word
            embedding[i + 2, :] = model[word]

        self.vocab_size = vocab_size + 2
        self.embedding_dim = embedding_dim
        self.word2id = word2id
        self.id2word = id2word
        self.embedding = embedding
        self.max_document_length = max_document_length

    def _text_transform(self, s, maxlen=None):
        if maxlen is None:
            maxlen = self.max_document_length
        if not isinstance(s, str):
            s = ""
        words = s.split()
        vec = []
        for w in words:
            if w in self.word2id:
                vec.append(self.word2id[w])
            else:
                vec.append(1)
        for i in range(len(words), maxlen):
            vec.append(0)
        return vec[:maxlen]
