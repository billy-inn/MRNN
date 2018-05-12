# ------------------------ PATH -------------------------

ROOT_PATH = "."
DATA_PATH = "%s/data" % ROOT_PATH
MPQA_PATH = "%s/database.mpqa.1.2" % DATA_PATH

LOG_DIR = "%s/log" % ROOT_PATH

# ------------------------ DATA -------------------------

ALL_DATA = "%s/all.tsv" % DATA_PATH
CLEAN_ALL_DATA = "%s/clean.tsv" % DATA_PATH

# EMBEDDING_DATA = "%s/glove.840B.300d.txt" % DATA_PATH
EMBEDDING_DATA = "%s/GoogleNews-vectors-negative300.bin" % DATA_PATH

# ------------------------ DATA -------------------------

MAX_DOCUMENT_LENGTH = 30

RANDOM_SEED = 2018
