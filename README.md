# MRNN

Attempt to re-implement "Modeling Compositionality with Multiplicative Recurrent Neural Networks": [paper](https://arxiv.org/abs/1412.6577).

**Problems**:

- After preprocessing, I only get 7870 phrases instead of 8022 reported in the paper.
- I'm skeptical about whether equation (5) can be simplified to equation (7) by adding bias units to x and h.

**Code**:

- `config.py`: configuration
- `data_preparer`: extract and preprocess the phrases
- `model.py`: code for the neural model
- `run.py`: run the experiments

**Results**:
