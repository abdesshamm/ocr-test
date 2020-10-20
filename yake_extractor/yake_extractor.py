
import numpy as np
import os
import re
import string

from segtok.segmenter import split_multi
from segtok.tokenizer import web_tokenizer, split_contractions


class YakeExtractor(object):
    """extract keyphrases from docs"""

    def __init__(self, lang="en", size_window=1,
                 max_n_gram=3, top_k=20, thresh=0.9):
        """
        Initialize the corpus.
        """
        super(YakeExtractor, self).__init__()
        self.lang = lang
        self.size_window = size_window
        self.max_n_gram = max_n_gram
        self.top_k = top_k
        self.thresh = thresh

        self.read_stop_words()
        self.ponctuation = set(string.punctuation)

        self.vocab = {}
        self.cooccur_left = {}
        self.cooccur_right = {}
        self.candidates = {}
        self.features = {}

    @staticmethod
    def text_to_sentences(text):
        """
        Convert text into sentences.
        """
        splited_text = [w for w in split_multi(text)]
        splited_sentences = [[w for w in split_contractions(
            web_tokenizer(s))] for s in splited_text]
        return [w for w in splited_sentences if len(w)]

    @staticmethod
    def is_digit(token):
        """
        Returns true if token is a digit
        """
        try:
            token_c = token.replace(",", ".")
            float(token_c)
            return True
        except:
            return False

    @staticmethod
    def is_acronym(token):
        """
        Returns true if token is a token.
        """
        nb_upper = len([c for c in token if c.isupper()])
        if nb_upper == len(token):
            return True
        return False

    @staticmethod
    def is_uppercase(token, idx):
        """
        Returns true if the token is a string.
        """
        if idx > 1 and len(token) > 1 and token[0].isupper():
            return True
        return False

    @staticmethod
    def is_unparsable_content(token, ponctuation):
        """
        Return true if token is unponctuation.
        """
        nb_digit = len([c for c in token if c.isdigit()])
        nb_alpha = len([c for c in token if c.isalpha()])
        nb_ponctuation = len([c for c in token if c in ponctuation])
        if (nb_ponctuation > 1) or (nb_digit > 0 and nb_alpha > 0) or (nb_digit == 0 and nb_alpha == 0):
            return True
        return False

    @staticmethod
    def extract_tag(token, idx, ponctuation):
        """
        Extracts the token from a token.
        """
        if YakeExtractor.is_digit(token):
            return "d"
        if YakeExtractor.is_acronym(token):
            return 'a'
        if YakeExtractor.is_uppercase(token, idx):
            return 'U'
        if YakeExtractor.is_unparsable_content(token, ponctuation):
            return 'u'
        return 'p'

    @staticmethod
    def levenshtein(seq1, seq2):
        """
        Compute the levenshtein distance between two sequences.
        """
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        dist = matrix[size_x - 1, size_y - 1]
        length = max(size_x - 1, size_y - 1)
        return 1 - float(dist) / length

    def read_stop_words(self):
        """
        Read stop stop words.
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        stop_words_path = os.path.join(
            dir_path,
            "stopwords",
            "stopwords_%s.txt" % self.lang)
        with open(stop_words_path, encoding='utf-8') as stop_fil:
            self.stop_words = set(stop_fil.read().lower().split("\n"))

    def build_vocab_coocurance_candidates(self, texts):
        """
        Parameters ---------- texts : list of sentences.
        """
        self.sentences = []

        for text in texts:
            text = text.replace('\n', ' ')
            text = text.replace('\t', ' ')
            self.sentences.extend(YakeExtractor.text_to_sentences(text))

        for id_sent, sent in enumerate(self.sentences):
            chunk = []
            for id_token, token in enumerate(sent):
                if len(token) == len([c for c in token if c in self.ponctuation]):
                    chunk = []
                else:
                    tag = YakeExtractor.extract_tag(
                        token, id_token, self.ponctuation)
                    lower_token = token.lower()

                    # is a stop word
                    for ponct in self.ponctuation:
                        no_ponct_lower_token = lower_token.replace(ponct, "")
                    is_stopword = (no_ponct_lower_token in self.stop_words) or len(
                        no_ponct_lower_token) <= 2

                    # update vocabulary
                    self.vocab.setdefault(lower_token, {"TF": 0,
                                                        "TF_a": 0,
                                                        "TF_U": 0,
                                                        "is_stopword": False,
                                                        "sentences_ids": [],
                                                        "docs_ids": []})

                    self.vocab[lower_token]["TF"] += 1
                    self.vocab[lower_token]["is_stopword"] = is_stopword
                    self.vocab[lower_token]["sentences_ids"].append(id_sent)

                    if tag == "a":
                        self.vocab[lower_token]["TF_a"] += 1
                    if tag == "U":
                        self.vocab[lower_token]["TF_U"] += 1

                    # update cooccur
                    self.cooccur_left.setdefault(lower_token, {})
                    self.cooccur_right.setdefault(lower_token, {})
                    if tag not in ["u", "d"]:
                        idx_window_min = max(0, len(chunk) - self.size_window)
                        idx_window_max = len(chunk)
                        for i in range(idx_window_min, idx_window_max):
                            if chunk[i]["tag"] not in ["u", "d"]:

                                self.cooccur_left[chunk[i]["lower_token"]
                                                  ].setdefault(lower_token, 0)
                                self.cooccur_left[chunk[i]
                                                  ["lower_token"]][lower_token] += 1
                                self.cooccur_right[lower_token
                                                   ].setdefault(chunk[i]["lower_token"], 0)
                                self.cooccur_right[lower_token][chunk[i]
                                                                ["lower_token"]] += 1

                    # update chunk
                    chunk.append({"lower_token": lower_token,
                                  "is_stopword": is_stopword,
                                  "tag": tag,
                                  "sent_position": id_sent})

                    # add candidates
                    idx_window_min = max(0, len(chunk) - (self.max_n_gram))
                    idx_window_max = len(chunk)
                    for i in range(idx_window_min, idx_window_max):
                        candidate = []
                        for j in range(i, idx_window_max):
                            candidate.append(chunk[j])
                        start_end_with_stopword = candidate[0]["is_stopword"] or candidate[-1]["is_stopword"]
                        start_with_digit = candidate[0]["tag"] == "d"
                        tag_u = len(
                            [token for token in candidate if token["tag"] == "u"])
                        tag_d = len(
                            [token for token in candidate if token["tag"] == "d"])
                        candidate_str = " ".join(
                            [token["lower_token"] for token in candidate])
                        if not start_with_digit and not start_end_with_stopword and (not tag_u or not tag_d):
                            self.candidates.setdefault(candidate_str, {})
                            self.candidates[candidate_str].setdefault("TF", 0)
                            self.candidates[candidate_str]["TF"] += 1

    def compute_features(self):
        """
        Compute features.
        """
        TF = [val["TF"] for val in self.vocab.values()]
        TF_nstopword = [val["TF"]
                        for val in self.vocab.values() if not val["is_stopword"]]
        mean_TF = np.mean(TF_nstopword)
        std_TF = np.std(TF_nstopword)
        max_TF = max(TF)

        for token in self.vocab.keys():

            self.features.setdefault(token, {})

            self.features[token]["CASING"] = max(
                self.vocab[token]["TF_a"], self.vocab[token]["TF_U"]) / (1 + np.log(self.vocab[token]["TF"]))

            self.features[token]["POSITION"] = np.log(
                np.log(3 + np.median(list(set(self.vocab[token]["sentences_ids"])))))

            self.features[token]["FREQUENCY"] = self.vocab[token]["TF"] / \
                (mean_TF + std_TF)

            WL = 0.0
            nb_link_left = float(len(list(self.cooccur_left[token].keys())))
            sum_link_left = float(
                np.sum(list(self.cooccur_left[token].values())))

            if sum_link_left:
                WL = nb_link_left / sum_link_left

            WR = 0.0
            nb_link_right = float(len(self.cooccur_right[token].keys()))
            sum_link_right = float(
                np.sum(list(self.cooccur_right[token].values())))
            if sum_link_right:
                WR = nb_link_right / sum_link_right

            self.features[token]["RELATEDNESS"] = 1 + \
                (WL + WR) * (self.vocab[token]["TF"] / max_TF)

            self.features[token]["DIFFERENT"] = float(
                len(set(self.vocab[token]["sentences_ids"]))) / len(self.sentences)

            CASE = self.features[token]["CASING"]
            POS = self.features[token]["POSITION"]
            NORM = self.features[token]["FREQUENCY"]
            REL = self.features[token]["RELATEDNESS"]
            SENT = self.features[token]["DIFFERENT"]

            self.features[token]["score"] = (
                (REL * POS) / (CASE + NORM / REL + SENT / REL))

    def compute_candidates_scores(self):
        """
        Parameters ---------- candidate scores.
        """
        for candidate_str in self.candidates.keys():
            tokens = candidate_str.split(" ")
            prod_S = 1.0
            sum_S = 0
            for i, token in enumerate(tokens):
                if self.vocab[token]["is_stopword"]:
                    left_token = tokens[i - 1]
                    right_token = tokens[i + 1]

                    try:
                        prob1 = float(
                            self.cooccur_left[left_token][token]) / self.vocab[left_token]["TF"]
                        prob2 = float(
                            self.cooccur_right[right_token][token]) / self.vocab[right_token]["TF"]
                        prob = prob1 * prob2
                    except:
                        prob = 0.0

                    prod_S *= (1 + (1 - prob))
                    sum_S -= (1 - prob)
                else:
                    prod_S *= self.features[token]["score"]
                    sum_S += self.features[token]["score"]

            tf = self.candidates[candidate_str]["TF"]
            self.candidates[candidate_str]["score"] = float(
                prod_S) / (tf * (1 + sum_S))

    def extract_keyphrases(self, docs):
        """
        Parameters ---------- docphr : dicts.
        """
        for doc in docs:
            self.build_vocab_coocurance_candidates(doc)
        self.compute_features()
        self.compute_candidates_scores()
        candidates_keyphrases = sorted([(cand_str, self.candidates[cand_str]["score"])
                                        for cand_str in self.candidates.keys()], key=lambda x: x[1])

        keyphrases = []
        for cand1 in candidates_keyphrases:
            add = True
            for cand2 in keyphrases:
                dist = YakeExtractor.levenshtein(cand1[0], cand2[0])
                if dist > self.thresh:
                    add = False
                    break
            if add:
                keyphrases.append((cand1[0], cand1[1]))
            if len(keyphrases) == self.top_k:
                break

        return keyphrases
