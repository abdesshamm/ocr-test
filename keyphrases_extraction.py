"""
KeyPhrasesExtractor extracts keyphrases from document using YAKE algorithm
Campos, R., Mangaravite, V., Pasquali, A., Jatowt, A., Jorge, A., Nunes, C. and Jatowt, A. (2020).
YAKE! Keyword Extraction from Single Documents using Multiple Local Features.
"""

import argparse
import itertools
from glob import glob
import numpy as np
import os

from difflib import SequenceMatcher
import numpy as np
import pandas as pd

from file_reader.file_reader import FilesReader
from yake_extractor.yake_extractor import YakeExtractor



class KeyPhrasesExtractor(object):
    """Extract keyphrases from documents
        input_directory       : directory where to find tags folders
                                (for each folder we need to find documents)
        output_name           : basename of the output  
        max_n_gram            : max n-gram of keyphrases
        top_k_by_doc          : number of k top keyprases to extract by document
        top_k_by_tag          : number of k top keyphrases to extract by tag
        clustering_keyohrases : clustering keyphrases
    """

    def __init__(self, input_directory, output_name, top_k_all_docs,
                 top_k_by_doc, top_k_by_tag,
                 max_n_gram, clustering_keyphrases):
        self.input_directory = input_directory
        self.output_name = output_name
        self.max_n_gram = max_n_gram
        self.top_k_all_docs = top_k_all_docs
        self.top_k_by_doc = top_k_by_doc
        self.top_k_by_tag = top_k_by_tag
        self.clustering_keyphrases = clustering_keyphrases
        self.file_reader = FilesReader()
        self.df_result = pd.DataFrame(
            columns=['lang', 'tag', 'filename', 'score', 'keyphrase'])

    @staticmethod
    def extract_keyphrases(text, max_n_gram, top_k, lang="en"):
        kw_extractor = YakeExtractor(lang=lang, top_k=top_k, max_n_gram=max_n_gram)
        keywords = kw_extractor.extract_keyphrases(text)
        return keywords

    @staticmethod
    def merge_documents_by_tag(texts):
        merged_docs = {}
        for lang in texts.keys():
            for tag in texts[lang].keys():
                if tag != "NO_TAG":
                    merged_docs.setdefault(lang, {})
                    merged_docs[lang][tag] = {"all_documents":
                                              list(texts[lang][tag].values())}
        return merged_docs

    @staticmethod
    def merge_documents(texts):
        merged_docs_by_tag = KeyPhrasesExtractor.merge_documents_by_tag(texts)
        res = {}
        for lang in merged_docs_by_tag.keys():
            res.setdefault(lang, {})
            res[lang].setdefault("all", {})
            res[lang]["all"].setdefault("all_documents", [])
            for tag in merged_docs_by_tag[lang].keys():
                res[lang]["all"]["all_documents"].extend(list(merged_docs_by_tag[lang][tag]["all_documents"]))
        return res

    @staticmethod
    def compute_similarity(phrase1, phrase2):
        tokens1 = phrase1.split(" ")
        tokens2 = phrase2.split(" ")
        return float(len(set(tokens1).intersection(
            set(tokens2)))) / max(len(tokens1), len(tokens2))

    @staticmethod
    def clustering_keyphrases(keyphrases):
        texts = keyphrases
        texts_value = []
        text_treated = []

        for conf1, text1 in texts:
            if text1 not in text_treated:
                text_treated.append(text1)
                res = [text1]
                confidence = conf1
                for conf2, text2 in texts:
                    if text2 not in text_treated:
                        if KeyPhrasesExtractor.compute_similarity(
                                text1, text2) > 0.25:
                            res.append(text2)
                            text_treated.append(text2)
                            confidence = min(confidence, conf2)
                texts_value.append((", \n ".join(res), confidence))
        return texts_value

    def extract_keyphrases_from_documents(self, documents, top_k, doc_by_doc=False):
        keyphrases = []
        for lang in documents.keys():
            for tag, docs in documents[lang].items():
                for filename, texts in docs.items():
                    if doc_by_doc:
                        texts = [texts]
                    file_keyphrases = KeyPhrasesExtractor.extract_keyphrases(
                        texts, self.max_n_gram, top_k, lang)
                    if self.clustering_keyphrases:
                        file_keyphrases = KeyPhrasesExtractor.clustering_keyphrases(
                            file_keyphrases)
                    for kw in file_keyphrases:
                        keyphrases.append([
                            lang, tag,
                            os.path.basename(filename),
                            kw[1],
                            kw[0]])
        return keyphrases

    def read_documents(self, filenames):
        self.file_reader.read_documents(filenames)

    def remove_documents(self, filenames):
        self.file_reader.remove_documents(filenames)

    def link_tag_to_document(self, filename, tag):
        self.file_reader.link_tag_to_document(filename, tag)

    def link_tag_to_documents(self, filename, tag):
        self.file_reader.link_tag_to_documents(filename, tag)

    def unlink_tag_from_document(self, filename, tag):
        self.file_reader.unlink_tag_from_document(filename, tag)

    def apply_doc_by_doc(self):
        documents = self.file_reader.get_documents()
        keyphrases = self.extract_keyphrases_from_documents(
            documents, self.top_k_by_doc, True)
        self.df_result = self.df_result.append(
            pd.DataFrame(keyphrases, columns=self.df_result.columns))

    def apply_by_tag(self):
        documents = self.file_reader.get_documents()
        merged_documents = KeyPhrasesExtractor.merge_documents_by_tag(
            documents)
        keyphrases = self.extract_keyphrases_from_documents(
            merged_documents, self.top_k_by_doc)
        self.df_result = self.df_result.append(
            pd.DataFrame(keyphrases, columns=self.df_result.columns))

    def apply_to_all_documents(self):
        documents = self.file_reader.get_documents()
        merged_documents = KeyPhrasesExtractor.merge_documents(documents)
        keyphrases = self.extract_keyphrases_from_documents(
            merged_documents, self.top_k_all_docs)
        self.df_result = self.df_result.append(
            pd.DataFrame(keyphrases, columns=self.df_result.columns))

    def normalize_scores(self):
        unique_value = self.df_result[[
            "lang", "tag", "filename"]].copy().drop_duplicates()

        self.df_result.index = range(len(self.df_result))
        for row in unique_value.values:
            lang, tag, filename = row
            ids = (self.df_result.lang == lang) & (self.df_result.tag ==
                                                   tag) & (self.df_result.filename == filename)

            self.df_result.score.loc[ids] = np.clip(
                  -np.log(self.df_result.score.loc[ids])*10,
                  0, 99)

    def save_results(self):
        self.normalize_scores()
        self.df_result.sort_values(
            by=['tag', 'filename', 'score'], inplace=True, ascending=False)
        self.df_result.to_csv("%s.csv" % self.output_name)

    def reset(self):
        self.file_reader = FilesReader()
        self.df_result = pd.DataFrame(
            columns=['lang', 'tag', 'filename', 'score', 'keyphrase'])

    def run(self):
        tag_directories = sorted(glob(os.path.join(self.input_directory, "*")))
        for tag_directory in tag_directories:
            tag = os.path.basename(tag_directory)
            doc_filenames = sorted(glob(os.path.join(tag_directory, "*")))
            self.read_documents(doc_filenames)
            self.link_tag_to_documents(doc_filenames, tag)
        self.apply_by_tag()
        self.apply_doc_by_doc()
        self.apply_to_all_documents()
        print(self.df_result)
        self.save_results()

def arguments():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument(
        'input_directory', help='directory where to find tags folders'
        '(for each folders we need to find documents).')
    parser.add_argument(
        'output_name', help='csv file where we put the final result.')
    parser.add_argument("--max_n_gram", help='max n-gram',
                        default=3, type=int)
    parser.add_argument('--top_k_all_docs', type=int, default=200,
                        help='extract k top keyphrases from all docs')
    parser.add_argument('--top_k_by_tag', type=int, default=100,
                        help='extract k top keyphrases by tag')
    parser.add_argument('--top_k_by_doc', type=int, default=50,
                        help='extract k top keyphrases')
    parser.add_argument("--clustering_keyphrases",
                        help="clustering keyphrases",
                        action="store_true")
    return parser.parse_args()


def main(args):
    KeyPhrasesExtractor(args.input_directory, args.output_name,
                        args.top_k_all_docs,
                        args.top_k_by_tag, args.top_k_by_doc,
                        args.max_n_gram,
                        args.clustering_keyphrases).run()


if __name__ == '__main__':
    args = arguments()
    main(args)
