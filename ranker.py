import math

# you can change whatever you want in this module, just make sure it doesn't
# break the searcher module
class Ranker:
    def __init__(self, indexer):
        self.indexer = indexer


    @staticmethod
    def rank_relevant_docs(relevant_docs, k=None):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param k: number of most relevant docs to return, default to everything.
        :param relevant_docs: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        ranked_results = sorted(relevant_docs.items(), key=lambda item: item[1], reverse=True)
        ranked_results_cut = []
        # clear all the low similarity retrieved docs to get higher precision.
        for idx, doc_tuple in enumerate(ranked_results):
            if doc_tuple[1][0] <= 0.1:
                ranked_results_cut = ranked_results[:idx]
                break
        if len(ranked_results_cut) > 0:
            ranked_results = ranked_results_cut
        if k is not None:
            ranked_results = ranked_results[:k]
        return [d[0] for d in ranked_results]

    def rank_tf_idf_query(self, query_as_dict, query_len):
        """
        calculate tf-idf for each term in query
        :param query_as_dict: {term : num of appearances in query}
        :param query_len: the length of the parsed query.
        :return:
        """
        query_term_weights_dict = {}
        for term in query_as_dict:
            # if term not in inverted index then this term has no impact on the retrieved docs.
            if term not in self.indexer.inverted_idx:
                continue
            tf = query_as_dict[term]/query_len
            idf = self.indexer.inverted_idx[term][1]
            w_iq = tf * idf
            query_term_weights_dict[term] = w_iq
        return query_term_weights_dict

    def calculate_cos_sim(self, query_term_weights_dict, term_in_docs_tuple_list, doc_id):
        """
        calculate cos-sim between doc and the query.
        :param query_term_weights_dict: {term : w_iq ...}
        :param term_in_docs_tuple_list: [(term, w_ij) ...]
        :param doc_id:
        :return: cos_sim and inner product
        """
        inner_product = 0
        segma_w_iq_pow = 0
        # calculate inner product
        for term in query_term_weights_dict:
            w_iq = query_term_weights_dict[term]
            for tuple_term in term_in_docs_tuple_list:
                if tuple_term[0] == term:
                    w_ij = tuple_term[1]
                    inner_product += w_ij * w_iq
                    break
            # calculation for the normalization of the cos-sim
            segma_w_iq_pow += math.pow(w_iq, 2)

        doc_sqrt_segma_wij_pow = self.indexer.weight_of_docs[doc_id]
        sqrt_segma_w_iq_pow = math.sqrt(segma_w_iq_pow)

        cos_sim_normalization = doc_sqrt_segma_wij_pow * sqrt_segma_w_iq_pow
        cos_sim = inner_product/cos_sim_normalization
        return cos_sim, inner_product

    def rank_combine(self, cos_sim, inner_product, max_inner_product):
        """
        calcute the similarity between query and doc with combination of methods to retrieve better results.
        :param cos_sim: the similarity in cos-sim method
        :param inner_product: the similarity in inner product method
        :param max_inner_product: the higher inner product similarity that we found for all docs.
        :return:
        """
        inner_product_between_0_1 = inner_product/max_inner_product
        rank = (cos_sim*0.8)
        rank += (inner_product_between_0_1*0.2)
        return rank



