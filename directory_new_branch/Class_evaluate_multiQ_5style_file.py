import json
import boto3
from tqdm import tqdm
from botocore.config import Config
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
from textblob import TextBlob



class FollowupQ_evaluation:
    def __init__(self, first_q_texts, first_r_texts, follow_q_texts):
        self.first_q_texts = first_q_texts
        self.first_r_texts = first_r_texts
        self.follow_q_texts = follow_q_texts


    def compute_mutual_information(self, text1, text2):
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


    def type_token_ratio(self, text):
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


    def compute_distinct_n(self, texts, n=1):
        total_ngrams = 0
        unique_ngrams = set()

        for text in texts:
            # Generate n-grams
            tokens = text.split()
            n_grams = list(ngrams(tokens, n))

            total_ngrams += len(n_grams)
            unique_ngrams.update(n_grams)

        return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0


    def get_multiple_score(self, first_q, first_r, follow_q_vec):
        len_tokens = 0
        MI_score = 0
        MI_score_R = 0
        TTR_score = 0
        Polarity_score = 0
        for val in follow_q_vec:
            tokens = nltk.word_tokenize(val[1:].lower())
            len_tokens += len(tokens)

            MI_score += self.compute_mutual_information(first_q, val[1:])
            MI_score_R += self.compute_mutual_information(first_r, val[1:])
            TTR_score += self.type_token_ratio(val[1:])

            blob = TextBlob(val)
            sent = 0
            for s in blob.sentences:

                sent += s.sentiment.polarity
            Polarity_score += sent / len(blob.sentences)


        len_tokens = len_tokens / len(follow_q_vec)
        MI_score = MI_score / len(follow_q_vec)
        MI_score_R = MI_score_R / len(follow_q_vec)
        TTR_score = TTR_score / len(follow_q_vec)

        distinct_1_across_multiple_Q = self.compute_distinct_n(follow_q_vec, n = 1)
        distinct_2_across_multiple_Q = self.compute_distinct_n(follow_q_vec, n = 2)

        Polarity_score = Polarity_score / len(follow_q_vec)
        return len_tokens, MI_score, MI_score_R, distinct_1_across_multiple_Q, distinct_2_across_multiple_Q, TTR_score, Polarity_score





if __name__ == "__main__":

    file_name = 'outputs_5together/m2_multiple_short_citation_5style_followup_QG_w_step_examples_multiQ_short.json'
    with open(file_name, "r", encoding="utf-8") as file:
        outputdata = json.load(file)

    for val in outputdata:
        # pull out generation
        first_q = outputdata[val]['first_question']
        first_r = outputdata[val]['first_response']
        followup_q = outputdata[val]['followup_question'][1:-1].split("', ")

        eval = FollowupQ_evaluation(first_q, first_r, followup_q)
        (len_tokens_basic_wo_example, MI_score_basic_wo_example, MI_score_basic_wo_example_R,
         distinct_1_across_multiple_Q_basic_wo_example, distinct_2_across_multiple_Q_basic_wo_example,
         TTR_score_basic_wo_example, Polarity_score) = eval.get_multiple_score(first_q, first_r, followup_q)
