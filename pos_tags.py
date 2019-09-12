import nltk
from collections import Counter
import word_importance
import data_loader
import definitions


def important_words_pos_summary(kv, topn, sim_norm=True, reverse=False):
    if sim_norm:
        _, words = word_importance.rank_by_sim_norm(kv, topn=topn, norm_ord=1, reverse=reverse)
    else:
        words = word_importance.rank_by_frequency(kv, topn=topn)
    # print(words)
    word_tag_pairs = [nltk.pos_tag([word], tagset='universal')[0][1] for word in words]
    # print(word_tag_pairs)
    print(Counter(word_tag_pairs))


if __name__ == "__main__":
    kv = data_loader.keyed_vec(definitions.GOOGLE_10000_KV_PATH)
    important_words_pos_summary(kv, 1000, sim_norm=False)
    important_words_pos_summary(kv, 1000, sim_norm=True)
    kv = data_loader.keyed_vec(definitions.GLOVE_10000_KV_PATH)
    important_words_pos_summary(kv, 1000, sim_norm=False)
    important_words_pos_summary(kv, 1000, sim_norm=True)