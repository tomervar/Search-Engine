import collections
import utils
import string
import os
import math

# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        self.inverted_idx = {}
        self.postingDict = {}
        self.terms_in_docs = {}
        self.weight_of_docs = {}
        self.config = config

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """

        document_dictionary = document.term_doc_dictionary
        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                # Update inverted index and posting
                if term not in self.inverted_idx.keys():
                    self.inverted_idx[term] = 1
                    self.postingDict[term] = []
                else:
                    self.inverted_idx[term] += 1

                f_ij = document_dictionary[term]
                tf_ij = f_ij / len(document_dictionary.keys())
                self.postingDict[term].append([document.tweet_id, document_dictionary[term], tf_ij, document.tweet_date])

            except:
                print('problem with the following key {}'.format(term[0]))
        self.terms_in_docs[document.tweet_id] = list(document_dictionary.keys())

    def add_idf_to_inverted_index(self, corpus_size):
        for term in self.inverted_idx.keys():
            d_ft = self.inverted_idx[term]
            idf_t = math.log2((corpus_size/d_ft))
            self.inverted_idx[term] = (d_ft, idf_t)

    def build_weight_of_docs(self):
        for tweet_id in self.terms_in_docs.keys():
            segma_w_ij_pow_of_doc = 0
            for term in self.terms_in_docs[tweet_id]:
                for list_in_posting in self.postingDict[term]:
                    if tweet_id == list_in_posting[0]:
                        tf = list_in_posting[2]
                        idf = self.inverted_idx[term][1]
                        tf_idf = tf * idf
                        list_in_posting.append(tf_idf)
                        tf_idf_pow = math.pow(tf_idf, 2)
                        segma_w_ij_pow_of_doc += tf_idf_pow
                        break

            sqrt_of_segma_w_ij_pow = math.sqrt(segma_w_ij_pow_of_doc)
            self.weight_of_docs[tweet_id] = sqrt_of_segma_w_ij_pow

        self.terms_in_docs = {}

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        raise NotImplementedError

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        raise NotImplementedError

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _is_term_exist(self, term):
        """
        Checks if a term exist in the dictionary.
        """
        return term in self.postingDict

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def get_term_posting_list(self, term):
        """
        Return the posting list from the index for a term.
        """
        return self.postingDict[term] if self._is_term_exist(term) else []
