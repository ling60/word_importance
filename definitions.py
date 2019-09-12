import os
from os.path import *

ROOT_DIR = dirname(abspath(__file__))  # Project Root
PAR_DIR = abspath(join(ROOT_DIR, os.pardir))
DATA_DIR = join(ROOT_DIR, "data")
RESULTS_DIR = join(ROOT_DIR, 'results')
EXTERNAL_DATA_DIR = join(PAR_DIR, "common-data")
FASTTEXT_DATA_DIR = join(EXTERNAL_DATA_DIR, "fasttext")
RAW_TOP5000_PATH = join(EXTERNAL_DATA_DIR, "top5000.txt")
TOP5000_LIST_PATH = join(DATA_DIR, "top5000_list")
GLOVE_COMMON_PREFIX = 'glove.840B.300d'
GLOVE_COMMON_KV_PATH = join(DATA_DIR, GLOVE_COMMON_PREFIX + '.kv')
GOOGLE_10000_PATH = join(EXTERNAL_DATA_DIR, "google-10000-english.txt")
GOOGLE_10000_KV_PATH = join(DATA_DIR, 'google-10000-english.kv')
# GLOVE_10000_KV_PATH = join(DATA_DIR, 'glove-10000-6b-300d.kv')
GLOVE_10000_KV_PATH = join(DATA_DIR, 'glove-10000-840b-300d.kv')
WIKI_NEWS_300_SUB_KV = join(DATA_DIR, "wiki-news-300d-sub.kv")

TEST_DATA_SETS_PATH = join(DATA_DIR, 'test_data_sets')
GOOGLE_ANALOGY_PATH = join(TEST_DATA_SETS_PATH, "google-analogy-test.txt")
RW_SIMILARITY_PATH = join(TEST_DATA_SETS_PATH, 'rare_words', 'rw.txt')
MEN_SIMILARITY_PATH = join(TEST_DATA_SETS_PATH, 'MEN', 'MEN_dataset_natural_form_full')
SimLex_SIMILARITY_PATH = join(TEST_DATA_SETS_PATH, 'SimLex-999', 'SimLex-999.txt')

