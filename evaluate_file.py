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



if __name__ == "__main__":

    file_name = 'outputs/single_short_citation_followup_QG_w_step_examples_short.json'
    with open(file_name, "r", encoding="utf-8") as file:
        outputdata = json.load(file)



    # MI evaluation
    q_len_dict = {}
    MI_score_dict = {}
    diversity_score_dict = {}

    for val in outputdata:

        # pull out generation
        first_q = outputdata[val]['first_question']
        first_r = outputdata[val]['first_response']
        if 'citation' in file_name:
            citations = outputdata[val]['citations']
        basic_wo_example = outputdata[val]['basic_wo_example']
        basic_w_example = outputdata[val]['basic_w_example']
        definition = outputdata[val]['definition']
        interpretation = outputdata[val]['interpretation']
        counterfactual = outputdata[val]['counterfactual']


        # basic length check
        q_len_dict['first_question'] = q_len_dict.get('first_question', 0) + len(nltk.word_tokenize(first_q.lower()))
        q_len_dict['len_basic_wo'] = q_len_dict.get('len_basic_wo', 0) + len(nltk.word_tokenize(basic_wo_example.lower()))
        q_len_dict['len_basic_w'] = q_len_dict.get('len_basic_w', 0) + len(nltk.word_tokenize(basic_w_example.lower()))
        q_len_dict['len_definition'] = q_len_dict.get('len_definition', 0) + len(nltk.word_tokenize(definition.lower()))
        q_len_dict['len_interpretation'] = q_len_dict.get('len_interpretation', 0) + len(nltk.word_tokenize(interpretation.lower()))
        q_len_dict['len_counterfactual'] = q_len_dict.get('len_counterfactual', 0) + len(nltk.word_tokenize(counterfactual.lower()))

        # MI check (init q w/ followup q)
        MI_score_dict['MI_basic_wo'] = MI_score_dict.get('MI_basic_wo', 0) + compute_mutual_information(first_q, basic_wo_example)
        MI_score_dict['MI_basic_w'] = MI_score_dict.get('MI_basic_w', 0) + compute_mutual_information(first_q, basic_w_example)
        MI_score_dict['MI_definition'] = MI_score_dict.get('MI_definition', 0) + compute_mutual_information(first_q, definition)
        MI_score_dict['MI_interpretation'] = MI_score_dict.get('MI_interpretation', 0) + compute_mutual_information(first_q, interpretation)
        MI_score_dict['MI_counterfactual'] = MI_score_dict.get('MI_counterfactual', 0) + compute_mutual_information(first_q, counterfactual)

        MI_score_dict['MI_basic_wo_R'] = MI_score_dict.get('MI_basic_wo_R', 0) + compute_mutual_information(first_r, basic_wo_example)
        MI_score_dict['MI_basic_w_R'] = MI_score_dict.get('MI_basic_w_R', 0) + compute_mutual_information(first_r, basic_w_example)
        MI_score_dict['MI_definition_R'] = MI_score_dict.get('MI_definition_R', 0) + compute_mutual_information(first_r, definition)
        MI_score_dict['MI_interpretation_R'] = MI_score_dict.get('MI_interpretation_R', 0) + compute_mutual_information(first_r, interpretation)
        MI_score_dict['MI_counterfactual_R'] = MI_score_dict.get('MI_counterfactual_R', 0) + compute_mutual_information(first_r, counterfactual)

        if 'citation' in file_name:
            MI_score_dict['MI_basic_wo_C'] = MI_score_dict.get('MI_basic_wo_C', 0) + compute_mutual_information(citations, basic_wo_example)
            MI_score_dict['MI_basic_w_C'] = MI_score_dict.get('MI_basic_w_C', 0) + compute_mutual_information(citations, basic_w_example)
            MI_score_dict['MI_definition_C'] = MI_score_dict.get('MI_definition_C', 0) + compute_mutual_information(citations, definition)
            MI_score_dict['MI_interpretation_C'] = MI_score_dict.get('MI_interpretation_C', 0) + compute_mutual_information(citations, interpretation)
            MI_score_dict['MI_counterfactual_C'] = MI_score_dict.get('MI_counterfactual_C', 0) + compute_mutual_information(citations, counterfactual)

        # diversity check (all followup q, style-aware followup q)
        across_all_diversity_vec = [basic_wo_example, basic_w_example, definition, interpretation, counterfactual]
        across_basic_diversity_vec = [basic_wo_example, basic_w_example]
        across_style_diversity_vec = [definition, interpretation, counterfactual]

        across_all_distinct_1 = compute_distinct_n(across_all_diversity_vec, n=1)
        across_all_distinct_2 = compute_distinct_n(across_all_diversity_vec, n=2)
        across_basic_distinct_1 = compute_distinct_n(across_basic_diversity_vec, n=1)
        across_basic_distinct_2 = compute_distinct_n(across_basic_diversity_vec, n=2)
        across_style_distinct_1 = compute_distinct_n(across_style_diversity_vec, n=1)
        across_style_distinct_2 = compute_distinct_n(across_style_diversity_vec, n=2)

        diversity_score_dict['all_distinct_1'] = diversity_score_dict.get('all_distinct_1', 0) + across_all_distinct_1
        diversity_score_dict['all_distinct_2'] = diversity_score_dict.get('all_distinct_2', 0) + across_all_distinct_2
        diversity_score_dict['basic_distinct_1'] = diversity_score_dict.get('basic_distinct_1', 0) + across_basic_distinct_1
        diversity_score_dict['basic_distinct_2'] = diversity_score_dict.get('basic_distinct_2', 0) + across_basic_distinct_2
        diversity_score_dict['style_distinct_1'] = diversity_score_dict.get('style_distinct_1', 0) + across_style_distinct_1
        diversity_score_dict['style_distinct_2'] = diversity_score_dict.get('style_distinct_2', 0) + across_style_distinct_2


        # TTR check
        avg_all_TTR = (type_token_ratio(basic_wo_example) + type_token_ratio(basic_w_example) + type_token_ratio(definition)
                       + type_token_ratio(interpretation) + type_token_ratio(counterfactual)) / 5
        avg_style_TTR = (type_token_ratio(definition) + type_token_ratio(interpretation)
                         + type_token_ratio(counterfactual)) / 3
        diversity_score_dict['avg_all_TTR'] = diversity_score_dict.get('avg_all_TTR', 0) + avg_all_TTR
        diversity_score_dict['avg_style_TTR'] = diversity_score_dict.get('avg_style_TTR', 0) + avg_style_TTR

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