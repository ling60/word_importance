import csv
import json
import logging
import os

import numpy as np
import scipy

import analogy_test
import data_loader
import definitions
import similarity_test
import word_importance
import words_deduction

top_words_dict = {'frequency': None,
                  'sim_l1_norm': None,
                  # 'sim_l2_norm': None
                  }
ranking_method_dict = {'frequency': 'fre',
                       'sim_l1_norm': 'sim_norm_L1',
                       'sim_l2_norm': 'sim_norm_L2'
                       }
# using RW as default.
data_loader_dict = {'sim': {
                        'MEN': data_loader.sim_men,
                         'simlex': data_loader.simlex_999,
                         'RW': data_loader.rare_words},
                    'analogy':
                        {'google': data_loader.google_analogy_test
                         }
                    }
methods = ["rotation"]

model_names = ['fast', 'glove']

kv_dict = {'fast10000': definitions.GOOGLE_10000_KV_PATH,
           'glove10000': definitions.GLOVE_10000_KV_PATH,
           'fast': definitions.WIKI_NEWS_300_SUB_KV,
           'glove': definitions.GLOVE_COMMON_KV_PATH}

func_dict = {'sim': similarity_test.load_and_evaluate,
             'analogy': analogy_test.load_and_evaluate}


def regression_for_path(path, p_threshold=0.05):
    score_dict = similarity_test.parse_result(path)
    print(path, score_dict['mean'])
    scores = np.asarray(score_dict['scores'])
    p_median = None
    p_mean = None
    try:
        pvalues = np.asarray(score_dict['pvals'])
        p_median = np.median(pvalues)
        p_mean = np.mean(pvalues)
        if p_threshold:
            scores = scores[pvalues <= p_threshold]
        print(len(scores))
    except TypeError:
        pass

    ranks = list(range(len(scores)))
    # res = scipy.stats.linregress(ranks, scores)
    # print("abs_mean:{}, slope:{}, p:{}".format(np.mean(np.abs(scores)), res.slope, res.pvalue))
    # print(scipy.stats.describe(scores))
    print("median of scores:{}, P: {}, p_mean:{}".format(np.median(scores), p_median, p_mean))
    print(scipy.stats.linregress(ranks, scores))
    return np.median(scores), p_median


def regression_test(topn, dataset_name, model_name, reverse, func_name, rand_n=0, p_threshold=0.05):
    result_dict = {}
    # if func_name == 'sim':
    #     methods = similarity_test.methods
    # elif func_name == 'analogy':
    #     methods = analogy_test.methods
    # else:
    #     raise NotImplementedError
    if topn < 0:
        kv = data_loader.keyed_vec(kv_dict[model_name])
        topn = len(kv.index2word)
    methods = similarity_test.methods
    for method in methods:
        r_dict = {}
        for key in top_words_dict.keys():
            path = make_result_path(topn, model_name, dataset_name, key, method, reverse=reverse, func_name=func_name, random_n=rand_n)
            score, p_median = regression_for_path(path, p_threshold)
            r_dict[key] = {'score': score,
                           'p_val': p_median}
        result_dict[method] = r_dict
    return result_dict


def batch_regression_test(topn=1000, func_name='sim'):
    if func_name == 'sim':
        assert len(similarity_test.methods) == 1
    dataset_names = list(data_loader_dict[func_name].keys())
    with open('{}_results.csv'.format(func_name), 'w', newline='') as f:
        res_writer = csv.writer(f)
        # res_writer.writerow([''] + dataset_names)

        for model_name in model_names:
            for dataset_name in dataset_names:
                res_writer.writerow(['', dataset_name])
                res_dict = regression_test(topn=topn, dataset_name=dataset_name, model_name=model_name,
                                           reverse=True, func_name=func_name)
                print(res_dict)
                # if func_name == 'sim':
                res_dict = res_dict[similarity_test.methods[0]]
                for sim_type in res_dict.keys():
                    # sim_type = list(res_dict.keys())[0]
                    type_str = ranking_method_dict[sim_type]
                    type_str = '{}-{}'.format(model_name, type_str)
                    for k, v in res_dict[sim_type].items():
                        res_writer.writerow(['{} {}'.format(type_str, k), v])
    print('finished')


def make_result_path(topn, model_name, dataset_name, top_words_method, test_method, reverse, func_name, random_n):
    if model_name == 'fast':  # default is fasttext
        fname = "top{}words-{}-{}-{}-{}.json".format(topn, func_name, dataset_name, top_words_method, test_method)
    else:
        fname = "top{}words-{}-{}-{}-{}-{}.json".format(topn, model_name, func_name, dataset_name, top_words_method,
                                                        test_method)
    if reverse:
        fname = "reverse_{}".format(fname)
    if random_n > 0:
        fname = "rand_{}_{}".format(random_n, fname)
    path = os.path.join(definitions.DATA_DIR, fname)
    return path


def evaluate_word(word, model_name, method, loader, evaluator):
    print("testing word:{}".format(word))
    # many functions could make inplace changes to given kv.
    # So we have to reload the file to ensure the data is unchanged.
    kv = data_loader.keyed_vec(kv_dict[model_name])
    new_kv = words_deduction.kv_deduct_word(kv, word, method=method)
    return evaluator(loader, new_kv)
    # X, y = loader()
    # # print(X.shape)
    # score, pval = evaluate_similarity(new_kv, X, y)
    # return score, pval


def batch_tests(topn, reverse, model_name, top_words_dict, eval_type, save=True, random_n=0):
    
    # todo: linear regression scores to see if the beta is negative and significant
    assert type(reverse) is bool
    kv_10000 = data_loader.keyed_vec(kv_dict[model_name + '10000'])
    if topn < 0:
        topn = len(kv_10000.index2word)
    top_words_by_frequency = word_importance.rank_by_frequency(kv_10000, topn=topn, reverse=reverse)

    indices, top_words_sim_l2_norm = word_importance.rank_by_sim_norm(kv_10000, topn=topn, reverse=reverse)
    _, top_words_sim_l1_norm = word_importance.rank_by_sim_norm(kv_10000, topn=topn, norm_ord=1,
                                                                reverse=reverse)

    if random_n > 0:
        assert topn > random_n
        rand_indices = np.sort(np.random.choice(range(topn), random_n, replace=False))
        top_words_sim_l2_norm = top_words_sim_l2_norm[rand_indices]
        top_words_sim_l1_norm = top_words_sim_l1_norm[rand_indices]
        top_words_by_frequency = top_words_by_frequency[rand_indices]
        print(top_words_sim_l1_norm)

    if 'frequency' in top_words_dict:
        top_words_dict['frequency'] = top_words_by_frequency
    if 'projection' in top_words_dict:
        top_words_dict['sim_l2_norm'] = top_words_sim_l2_norm
    if 'sim_l1_norm' in top_words_dict:
        top_words_dict['sim_l1_norm'] = top_words_sim_l1_norm
    for dataset_name, loader in data_loader_dict[eval_type].items():
        for key, top_words in top_words_dict.items():
            for method in methods:
                score_dict = {}
                scores = []
                pvals = []
                path = make_result_path(topn, model_name, dataset_name, key, method, reverse=reverse,
                                        func_name=eval_type, random_n=random_n)
                print("path to save:{}".format(path))
                for word in top_words:
                    score, pval = evaluate_word(word, model_name, method=method, loader=loader,
                                                evaluator=func_dict[eval_type])
                    print("score:{}, p:{}".format(score, pval))
                    scores.append(score)
                    pvals.append(pval)
                mean = np.mean(np.asarray(scores))
                print("final:{}".format(mean))
                score_dict['mean'] = mean
                score_dict['scores'] = scores
                score_dict['pvals'] = pvals
                if save:
                    logging.info('saving to {}...'.format(path))
                    with open(path, 'w') as f:
                        json.dump(score_dict, f)


if __name__ == "__main__":
    # batch_regression_test(topn=100, func_name='analogy')
    # dataset_name = 'RW'
    # loader = similarity_test.data_loader_dict[dataset_name]
    # x, y = loader()
    # full_kv = data_loader.keyed_vec(definitions.WIKI_NEWS_300_SUB_KV)
    # print(similarity_test.evaluate_similarity(full_kv, x, y))
    # path = 'data/rand_200_top9915words-sim-RW-frequency-rotation.json'
    # regression_for_path(path)
    # regression_test(-1, dataset_name, 'fast', True, func_name='sim', rand_n=200)

    batch_tests(-1, reverse=False, model_name='fast', top_words_dict=top_words_dict, eval_type='sim', save=True, random_n=1000)

