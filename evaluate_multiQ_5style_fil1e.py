import json
import boto3
from tqdm import tqdm
from botocore.config import Config
import pandas as pd

import pdb

from collections import Counter

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import string

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')



def compute_mutual_information(text1, text2):

    # tokenize
    tokens1 = nltk.word_tokenize(text1.lower())
    tokens2 = nltk.word_tokenize(text2.lower())

    # get unique vocabulary
    vocab = list(set(tokens1) | set(tokens2))

    # word frequency
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform([text1, text2]).toarray()

    # compute MI
    mi_score = mutual_info_score(X[0], X[1])

    return mi_score



def type_token_ratio(text):
    # tokenize
    tokens = word_tokenize(text)

    # remove punctuation & stop words
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # calculate the number of unique token
    types = set(tokens)
    num_types = len(types)
    num_tokens = len(tokens)

    # calculate TTR
    ttr = num_types / num_tokens if num_tokens > 0 else 0
    return ttr


def compute_distinct_n(texts, n=1):

    total_ngrams = 0
    unique_ngrams = set()

    for text in texts:
        # Generate n-grams
        tokens = text.split()
        n_grams = list(ngrams(tokens, n))

        total_ngrams += len(n_grams)
        unique_ngrams.update(n_grams)

    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0


def get_multiple_score(first_q, first_r, follow_q_vec):
    len_tokens = 0
    MI_score = 0
    MI_score_R = 0
    TTR_score = 0
    for val in follow_q_vec:
        tokens = nltk.word_tokenize(val[1:].lower())
        len_tokens += len(tokens)

        MI_score += compute_mutual_information(first_q, val[1:])
        MI_score_R += compute_mutual_information(first_r, val[1:])
        TTR_score += type_token_ratio(val[1:])

    len_tokens = len_tokens / len(follow_q_vec)
    MI_score = MI_score / len(follow_q_vec)
    MI_score_R = MI_score_R / len(follow_q_vec)
    TTR_score = TTR_score / len(follow_q_vec)

    distinct_1_across_multiple_Q = compute_distinct_n(follow_q_vec, n = 1)
    distinct_2_across_multiple_Q = compute_distinct_n(follow_q_vec, n = 2)

    return len_tokens, MI_score, MI_score_R, distinct_1_across_multiple_Q, distinct_2_across_multiple_Q, TTR_score

if __name__ == "__main__":

    file_name = 'outputs_5together/multiple_short_citation_5style_followup_QG_w_step_examples_multiQ_short.json'
    with open(file_name, "r", encoding="utf-8") as file:
        outputdata = json.load(file)



    # MI evaluation
    q_len_dict = {}
    MI_score_dict = {}
    diversity_score_dict = {}

    c = 0
    for val in outputdata:
        # pdb.set_trace()
        # pull out generation
        first_q = outputdata[val]['first_question']
        first_r = outputdata[val]['first_response']
        followup_q = outputdata[val]['followup_question'][1:-1].split("', ")


        (len_tokens_basic_wo_example, MI_score_basic_wo_example, MI_score_basic_wo_example_R,
         distinct_1_across_multiple_Q_basic_wo_example, distinct_2_across_multiple_Q_basic_wo_example,
         TTR_score_basic_wo_example) = get_multiple_score(first_q, first_r, followup_q)


        q_len_dict['len_basic_wo'] = q_len_dict.get('len_basic_wo', 0) + len_tokens_basic_wo_example


        MI_score_dict['MI_basic_wo'] = MI_score_dict.get('MI_basic_wo', 0) + MI_score_basic_wo_example


        MI_score_dict['MI_basic_wo_R'] = MI_score_dict.get('MI_basic_wo_R', 0) + MI_score_basic_wo_example_R


        diversity_score_dict['basic_wo_distinct_1'] = diversity_score_dict.get('basic_wo_distinct_1',
                                                                               0) + distinct_1_across_multiple_Q_basic_wo_example


        diversity_score_dict['basic_wo_TTR'] = diversity_score_dict.get('basic_wo_TTR', 0) + TTR_score_basic_wo_example


    q_len_dict = {key: value / len(outputdata) for key, value in q_len_dict.items()}
    MI_score_dict = {key: value / len(outputdata) for key, value in MI_score_dict.items()}
    diversity_score_dict = {key: value / len(outputdata) for key, value in diversity_score_dict.items()}

print("length:")
print(q_len_dict)
print()
print("MI:")
print(MI_score_dict)
print()
print("diversity:")
print(diversity_score_dict)