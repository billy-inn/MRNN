import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from utils import data_utils

class Model(nn.Module):
    def __init__(self, seq_len, num_classes, pretrained_embedding, hparams, use_cuda):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.use_cuda = use_cuda
        self.d = 25

        vocab_size = pretrained_embedding.shape[0]
        embedding_dim = pretrained_embedding.shape[1]

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.embeddings.weight = nn.Parameter(torch.FloatTensor(pretrained_embedding))
        self.embeddings.weight.requires_grad = False

        self.A = nn.Parameter(torch.FloatTensor(self.d, embedding_dim, self.d))
        self.output = nn.Linear(self.d, num_classes)

    def init_hidden(self, batch_size):
        h0 = autograd.Variable(torch.zeros(batch_size, self.d))
        return h0.cuda() if self.use_cuda else h0

    def cell(self, x, h_prev):
        h = torch.matmul(x, self.A)
        h = torch.matmul(h.view(self.d, -1, 1, self.d), h_prev.unsqueeze(2))
        h = F.tanh(torch.t(h.squeeze()))
        return h

    def forward(self, words, batch_size):
        wrd_embeds = self.embeddings(words)
        h = self.init_hidden(batch_size)
        for i in range(self.seq_len):
            h = self.cell(wrd_embeds[:, i, :], h)
        logits = self.output(h)
        return F.sigmoid(logits)

class Classifier:
    def __init__(self, seq_len, num_classes, pretrained_embedding, hparams):
        self.num_classes = num_classes
        self.batch_size = hparams["batch_size"]
        self.num_epochs = hparams["num_epochs"]
        lr = hparams["lr"]
        l2_reg_lambda = hparams["l2_reg_lambda"]

        self.use_cuda = torch.cuda.is_available()
        self.model = Model(
            seq_len, num_classes, pretrained_embedding, hparams, self.use_cuda)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad,
            self.model.parameters()), lr=lr, weight_decay=l2_reg_lambda)
        self.loss_fn = nn.BCELoss()
        if self.use_cuda:
            self.model = self.model.cuda()

    def init(self):
        def init_weights(model):
            if type(model) in [nn.Linear]:
                nn.init.xavier_normal(model.weight.data)

        self.model.apply(init_weights)

    def _variable(self, data):
        data = np.array(data)
        data = autograd.Variable(torch.from_numpy(data))
        return data.cuda() if self.use_cuda else data

    def fit(self, train, valid=None):
        batches = data_utils.batch_iter(train, self.batch_size, self.num_epochs)
        data_size = len(train)
        num_batches_per_epoch = int((data_size - 1) / self.batch_size) + 1
        best_valid_loss = 1e5
        best_valid_epoch = 0
        step = 0
        for batch in batches:
            step += 1
            words_batch, labels_batch = zip(*batch)
            batch_size = len(words_batch)

            self.model.zero_grad()

            words = self._variable(words_batch)
            labels = self._variable(labels_batch)

            probs = self.model(words, batch_size)
            loss = self.loss_fn(probs, labels)
            time_str = datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss.data[0]))

            loss.backward()
            self.optimizer.step()

            if (step % num_batches_per_epoch == 0) and (valid is not None):
                print("\nValidation:")
                print("previous best valid loss {:g} at epoch {}".format(
                    best_valid_loss, best_valid_epoch))
                rloss = self.get_rank_loss(valid)
                print("epoch: {}, loss {:g}".format(
                    step // num_batches_per_epoch, rloss))
                print("")
                if rloss < best_valid_loss:
                    best_valid_loss = rloss
                    best_valid_epoch = step // num_batches_per_epoch
                if step // num_batches_per_epoch - best_valid_epoch > 3:
                    break
        return best_valid_epoch, best_valid_loss

    def get_score(self, v):
        for i in range(4):
            if v[i] < 0.5:
                return i
        return 4

    def get_rank_loss(self, test):
        batches = data_utils.batch_iter(test, self.batch_size, 1, shuffle=False)
        cnt = 0.0
        total_loss = 0.0
        for batch in batches:
            words_batch, labels_batch = zip(*batch)
            batch_size = len(words_batch)

            words = self._variable(words_batch)

            probs = self.model(words, batch_size)
            if self.use_cuda:
                probs = probs.cpu()
            probs = probs.data.numpy()
            cnt += batch_size
            for i in range(batch_size):
                s1 = self.get_score(probs[i])
                s2 = self.get_score(labels_batch[i])
                print(s1, s2)
                total_loss += abs(s1 - s2)
        return total_loss / cnt
