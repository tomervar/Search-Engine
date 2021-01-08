from ranker import Ranker
import utils
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import lin_thesaurus as thes
from nltk import pos_tag
# nltk.download('averaged_perceptron_tagger')
# nltk.download('lin_thesaurus')
# nltk.download('wordnet')
from spellchecker import SpellChecker
from nltk.corpus import wordnet


# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer
        self._ranker = Ranker(indexer)
        self._model = model
        self.with_thesaurus = False
        self.with_spelling_correction = False
        self.with_wordNet = False

    def set_thesaurus(self):
        self.with_thesaurus = True

    def set_spelling_correction(self):
        self.with_spelling_correction = True

    def set_wordNet(self):
        self.with_wordNet = True

    def spelling_correction_checker(self, query_as_list):
        spell = SpellChecker()
        for idx, term in enumerate(query_as_list):
            if term not in self._indexer.inverted_idx:
                misspelled = spell.unknown([term])
                for word in misspelled:
                    correct_word = spell.correction(word)
                    if correct_word in self._indexer.inverted_idx:
                        query_as_list[idx] = correct_word


    def build_wordNet_for_query(self,query_as_dict):
        list_of_terms = list(query_as_dict.keys())
        for idx, term in enumerate(list_of_terms):
            if idx % 2 != 0:
                continue
            syn = True
            ant = True
            for synset in wordnet.synsets(term):
                for lemma in synset.lemmas():
                    if syn:
                        lemma_name = lemma.name()
                        if not lemma_name.startswith(term):
                            if lemma_name.upper() in self._indexer.inverted_idx:
                                lemma_name = lemma_name.upper()
                            if lemma_name in self._indexer.inverted_idx:
                                if lemma_name in query_as_dict:
                                    query_as_dict[lemma_name] += 0.5
                                else:
                                    query_as_dict[lemma_name] = 0.5
                                syn = False
                    lemma_antonyms = lemma.antonyms()
                    if ant and lemma_antonyms:
                        lemma_antonyms_name = lemma_antonyms[0].name()
                        if lemma_antonyms_name.upper() in self._indexer.inverted_idx:
                            lemma_antonyms_name = lemma_antonyms_name.upper()
                        if lemma_antonyms_name in self._indexer.inverted_idx:
                            if lemma_antonyms_name in query_as_dict:
                                query_as_dict[lemma_antonyms_name] += 0.4
                            else:
                                query_as_dict[lemma_antonyms_name] = 0.4
                            ant = False

                if not ant and not syn:
                    break

    def build_thesaurus_for_query(self, query_as_dict, query):
        list_of_terms = list(query_as_dict.keys())
        # list_of_terms = word_tokenize(query)
        parts_of_speech_tags = pos_tag(list_of_terms)
        for word, word_type in parts_of_speech_tags:
            if word_type == "NN":
                list_from_thes = list(thes.synonyms(word, fileid="simN.lsp"))
            elif word_type == "VB":
                list_from_thes = list(thes.synonyms(word, fileid="simV.lsp"))
            elif word_type == "JJ":
                list_from_thes = list(thes.synonyms(word, fileid="simA.lsp"))
            else:
                continue

            if len(list_from_thes) > 0:
                word_from_thes = list_from_thes[0]
                if word_from_thes.upper() in self._indexer.inverted_idx:
                    word_from_thes = word_from_thes.upper()
                if word_from_thes in self._indexer.inverted_idx:
                    if word_from_thes in query_as_dict:
                        query_as_dict[word_from_thes] += 0.4
                    else:
                        query_as_dict[word_from_thes] = 0.4

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        query_as_list = self._parser.parse_sentence(query)

        if self.with_spelling_correction:
            self.spelling_correction_checker(query_as_list)

        query_as_dict = {}
        for term in query_as_list:
            if term.upper() in self._indexer.inverted_idx:
                term = term.upper()
            if term in self._indexer.inverted_idx:
                if term in query_as_dict:
                    query_as_dict[term] += 1
                else:
                    query_as_dict[term] = 1

        query_len_before_thes = len(query_as_dict)

        if self.with_thesaurus:
            self.build_thesaurus_for_query(query_as_dict, query)

        if self.with_wordNet:
            self.build_wordNet_for_query(query_as_dict)

        relevant_docs = self._relevant_docs_from_posting(query_as_dict, len(query_as_list), query_len_before_thes)
        n_relevant = len(relevant_docs)
        ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs, k)
        return n_relevant, ranked_doc_ids

    # feel free to change the signature and/or implementation of this function 
    # or drop altogether.
    def _relevant_docs_from_posting(self, query_as_dict, query_len, query_len_before_thes):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param query_as_list: parsed query tokens
        :return: dictionary of relevant documents mapping doc_id to document frequency.
        """
        relevant_docs = {}
        relevant_docs_with_weight = {}
        query_term_weights_dict = self._ranker.rank_tf_idf_query(query_as_dict, query_len)
        for term in query_term_weights_dict:
            posting_list = self._indexer.get_term_posting_list(term)
            for list_in_posting_list in posting_list:
                doc_id = list_in_posting_list[0]
                w_ij = list_in_posting_list[4]
                doc_date = list_in_posting_list[3]
                if doc_id in relevant_docs_with_weight:
                    relevant_docs_with_weight[doc_id][1].append((term, w_ij))
                else:
                    relevant_docs_with_weight[doc_id] = [doc_date, [(term, w_ij)]]
        doc_id_to_cosine_and_inner = {}
        max_inner_product = 0
        for doc_id in relevant_docs_with_weight:
            if len(relevant_docs_with_weight[doc_id][1]) > (query_len_before_thes*(40/100)):
                cos_sim, inner_product = self._ranker.calculate_cos_sim(query_term_weights_dict, relevant_docs_with_weight[doc_id][1], doc_id)
                doc_id_to_cosine_and_inner[doc_id] = (cos_sim, relevant_docs_with_weight[doc_id][0], inner_product)
                if max_inner_product < inner_product:
                    max_inner_product = inner_product
            elif query_len_before_thes < 4:
                cos_sim, inner_product = self._ranker.calculate_cos_sim(query_term_weights_dict, relevant_docs_with_weight[doc_id][1], doc_id)
                doc_id_to_cosine_and_inner[doc_id] = (cos_sim, relevant_docs_with_weight[doc_id][0], inner_product)
                if max_inner_product < inner_product:
                    max_inner_product = inner_product
            # cos_sim = self._ranker.calculate_cos_sim(query_term_weights_dict, relevant_docs_with_weight[doc_id][1], doc_id)
            # relevant_docs[doc_id] = (cos_sim, relevant_docs_with_weight[doc_id][0])
        for doc_id in doc_id_to_cosine_and_inner:
            tup = doc_id_to_cosine_and_inner[doc_id]
            cos_sim = tup[0]
            date = tup[1]
            inner_product = tup[2]
            rank = self._ranker.rank_combine(cos_sim, inner_product, max_inner_product)
            relevant_docs[doc_id] = (rank, date)
        return relevant_docs
