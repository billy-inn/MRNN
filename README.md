# MRNN

Attemption to re-implement "Modeling Compositionality with Multiplicative Recurrent Neural Networks" ([paper](https://arxiv.org/abs/1412.6577)).

**Problems**:

- After preprocessing, I only get 7870 phrases instead of 8022 reported in the paper.
- Due to the size of the word embeddings, I use 300d GloVe which is easier to download on a server instead of 300d Word2Vec.
- I'm skeptical about whether equation (5) can be simplified to equation (7) by adding bias units to x and h.
- Missing some important implementation details, *e.g.*, whether to employ regularization and what's it; which optimization method is used; what's the learning rate; how many epochs to run or whether to use early stopping and etc.
- From prelimenary experiments, I didn't see any boost on the performance by adding the multiplicative term. And I don't think this can be changed by tuning hyperparameters.

**Code**:

- `config.py`: configuration
- `data_preparer`: extract and preprocess the phrases
- `model.py`: code for the neural model
- `run.py`: run the experiments
- Run the experiments: `python run.py <mode>` mode can be *mRNN7* (eq. 7), *mRNN5* (eq.5) and *RNN* (vanilla rnn).

**Prelimenary Results**:

Due to the time constraint, I run the experiments with a 9:1 train/validation split with 200 epochs and report the best validation results. Although it will give over-optimistic results in practice, the results are still nowhere near the results reported in the paper.

- Cannot get meaningful validation results with equation (5)
- With equation (7), it can get better results (~0.59).
