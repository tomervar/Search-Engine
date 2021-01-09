from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from document import Document

from stemmer import Stemmer
import re


class Parse:

    def __init__(self, with_stemmer=False):
        # additional stop words.
        self.our_stop_words = [".", "y'all", "didn't", "here's", "don't", "would", "oh", "etc", "i'd",
                               "can't", "wouldn't", "that's", "via", "let's", "i've", "he's", "it'll", "aka",
                               "we've", "due", "i'm", "rt", "tr"]
        self.stop_words = stopwords.words('english') + self.our_stop_words
        # dictionary to mark all the words suspected to save with upper, for all the corpus.
        # true means we didnt see this word with lower case. else false.
        self.upper_case_dict = {}
        # dictionary to mark all the entities, for all the corpus.
        # the value is the number of tweets that have this entitie
        self.names_and_entities = {}
        # save the value of number to help us handle all the cases that contains numbers.
        self.term_of_num = ""
        # save the value of term if first letter is upper case.
        self.term_of_entitie = ""
        # save all the entities for the current tweet.
        self.tweet_entities = []
        # list for all punctuation that we want to remove
        self.punctuation = [".", ",", "-", "_", ":", ";", "!", "?", "(", ")", "[", "]", "{", "}", "'", '"',
                            "&", "~", "/", "=", "+", "|", "^", "*", "<", ">", "`"]
        self.with_stemmer = with_stemmer
        self.stemmer = Stemmer()
        # dict of months to change the date in the documents.
        self.month_dict = {
                    'Jan': "01",
                    'Feb': "02",
                    'Mar': "03",
                    'Apr': "04",
                    'May': "05",
                    'Jun': "06",
                    'Jul': "07",
                    'Aug': "08",
                    'Sep': "09",
                    'Oct': "10",
                    'Nov': "11",
                    'Dec': "12"
                }

    def parse_sentence(self, text):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text:
        :return:
        """
        t = RegexpTokenizer('\s+', gaps=True)  # split with spaces.

        text_tokens = t.tokenize(text)
        text_tokens_without_stopwords = []  # the list that we will return to search_engine
        for term in text_tokens:
            while len(term) > 0 and (term[0] in self.punctuation or not term[0].isascii()):
                term = term[1:]

            if len(term) > 0 and (term[-1] in self.punctuation or not term[-1].isascii()):
                while len(term) > 0 and (term[-1] in self.punctuation or not term[-1].isascii()):
                    term = term[:-1]
                terms_after_rules = self.rule_checking(term)
                terms_after_rules += self.rule_checking(".")

            else:
                terms_after_rules = self.rule_checking(term)  # send to funcation that check all parser rules

            text_tokens_without_stopwords.extend(terms_after_rules)
        if len(self.term_of_num) > 0 or len(self.term_of_entitie) > 0:
            text_tokens_without_stopwords.extend(self.rule_checking("."))
        # check if we saw the entities over the corpus.
        for entitie in self.tweet_entities:
            if entitie in self.names_and_entities:
                self.names_and_entities[entitie] += 1  # increase the number of tweets that we saw this entitie
            else:
                self.names_and_entities[entitie] = 1  # we saw the enttie for the first time

        # resets for the next tweet
        self.term_of_num = ""
        self.term_of_entitie = ""
        self.tweet_entities = []

        return text_tokens_without_stopwords

    def rule_checking(self, term):
        """
        in this function we check all the rules for single term
        :param term: string from tweet
        :return: list of terms following the rules
        """
        text_tokens_without_stopwords = []
        term_lower = term.lower()
        # find patterns of urls and term with number'$' or '$'number
        pattern_url = re.match(r'(https?)?(?:://)?(www)?\.?(\w+\.\w+(?:\.\w+)*)', term)

        # check if the term isn't a stop_word then we'll parse it.
        if term_lower not in self.stop_words and (len(term_lower) > 1 or term_lower.isdigit()) and term_lower.isascii():
            # if the previous term was number
            if len(self.term_of_num) > 0:
                # send it to function that transform numbers according to the rules.
                list_of_number_to_add = self.numbers_handling(term, term_lower)
                self.term_of_num = ""  # reset the number's term
                text_tokens_without_stopwords.extend(list_of_number_to_add)

            # if the previous term started with capital letter
            elif len(self.term_of_entitie) > 0:
                # if the current term started with capital letter
                if term[0].isupper():
                    self.term_of_entitie += " "+term  # concatenate the current term to the right entitie.
                else:
                    # handle the term_of_entitie that is the full version entitie
                    list_of_entitie = self.handle_entitie()
                    text_tokens_without_stopwords.extend(list_of_entitie)
                    self.term_of_entitie = ""
                    # send the current term to rule checking and add it to the returned list
                    text_tokens_without_stopwords.extend(self.rule_checking(term))

            # if the term starts with digit and not contains '$'.
            elif term[0].isdigit() or term[0] == "$":
                # check if all the term is numeric
                if term.replace('.', '', 1).replace(',', '').isdigit():
                    if re.match(r'^\d+(\,\d+)*(\.\d+)?$', term) is not None:
                        self.term_of_num = term.replace(',', '')  # remove commas and save the number for next iteration.
                    else:
                        text_tokens_without_stopwords.append(term_lower)
                elif re.match(r'\$\d+$|\d+\$$|\$\d+\.\d+$|\d+\.\d+\$$|\$\d+(\,\d+)*(\.\d)?(\d+)?$|\d+(\,\d+)*(\.\d)?(\d+)?\$$', term) is not None:
                    term = term.replace('$', '').replace(',', '')  # transform the term to numbers only
                    string_of_dollar = self.transform_number(float(term))  # transform the number according to rules.
                    string_of_dollar += "$"
                    text_tokens_without_stopwords.append(string_of_dollar)

                # send it to function that can handle the mix of numbers and else.
                else:
                    list_of_number_to_add = self.numbers_handling(term, term_lower)
                    text_tokens_without_stopwords.extend(list_of_number_to_add)

            # handle tags
            elif term.startswith("@") and len(term) > 1:
                # clear all the repeating tags
                last_tag = 0
                for idx, ch in enumerate(term):
                    if term[idx] == "@":
                        last_tag = idx
                term = term[last_tag:]
                if len(term) > 1:
                    text_tokens_without_stopwords.append(term)

            # if hashtag
            elif term.startswith("#") and len(term) > 1:
                # clear all the repeating hashtags
                last_hashtag = 0
                for idx, ch in enumerate(term):
                    if term[idx] == "#":
                        last_hashtag = idx
                term = term[last_hashtag:]
                if len(term) > 1:
                    hashtag_list = self.split_hashtags(term)  # function that split the hashtag accrding to the rules.
                    text_tokens_without_stopwords.extend(hashtag_list)

            # if url was found
            elif pattern_url is not None:
                list_of_url = self.url_handler(pattern_url, term)  # send to function that split the url.
                text_tokens_without_stopwords.extend(list_of_url)

            elif "/" in term:
                list_of_terms = term.split("/")
                for subterm in list_of_terms:
                    text_tokens_without_stopwords.extend(self.rule_checking(subterm))

            # if the first letter is capital
            elif term[0].isupper():

                self.term_of_entitie = term  # save it because it can be an entitie or the start of one.

            # if no specific rule was found
            else:
                if self.with_stemmer:
                    term_lower = self.stemmer.stem_term(term)
                self.upper_case_dict[term_lower] = False  # add to dictionary and nerrow it down as a name or entitie.
                text_tokens_without_stopwords.append(term_lower)

        # the word is stop word and the word before started with capital letter
        else:
            if len(self.term_of_entitie) > 1:
                terms_lower = self.handle_entitie()
                text_tokens_without_stopwords.extend(terms_lower)
                self.term_of_entitie = ""

            if len(self.term_of_num) > 0:
                list_num_terms = self.numbers_handling(term, term_lower)
                text_tokens_without_stopwords.extend(list_num_terms)
                self.term_of_num = ""

        return text_tokens_without_stopwords

    def numbers_handling(self, term, term_lower):
        """
        the function will classify the pattern of the term and handle it accordingly
        :param term: the current term
        :param term_lower: the term in lower case.
        :return: list of terms
        """
        list_of_terms = []
        # catch patterns of numbers that end with k/m/b
        pattern_k = re.match(r'\d+\.\d+k$|\d+k$',term_lower)
        pattern_m = re.match(r'\d+\.\d+m$|\d+m$',term_lower)
        pattern_b = re.match(r'\d+\.\d+b$|\d+b$',term_lower)

        # handle case "number %" or "number percent(age)"
        if term_lower == "%" or term_lower.startswith("percent"):
            list_of_terms.append(self.term_of_num+"%")

        # handle case "number $" or "number dollar(s)"
        elif term_lower == "$" or term_lower.startswith("dollar"):
            list_of_terms.append(self.term_of_num + "$")

        # handle case "number k" or "number thousand(s)"
        elif term_lower == "k" or term_lower.startswith("thousand"):
            num_term = self.transform_number(float(self.term_of_num) * 1000)
            list_of_terms.append(num_term)

        # handle case "number m" or "number million(s)"
        elif term_lower == "m" or term_lower.startswith("million"):
            num_term = self.transform_number(float(self.term_of_num) * pow(1000, 2))
            list_of_terms.append(num_term)

        # handle case "number b" or "number billion(s)"
        elif term_lower == "b" or term_lower.startswith("billion"):
            num_term = self.transform_number(float(self.term_of_num) * pow(1000, 3))
            list_of_terms.append(num_term)

        # handle case "number fraction" or "fraction"
        elif term.replace('/', '', 1).replace('\\', '').replace(',', '').isdigit() and len(self.term_of_num) > 0:
            num_term = self.transform_number(float(self.term_of_num))
            if term.replace(',', '').isdigit():
                term = self.transform_number(float(term.replace(',', '')))
                list_of_terms.extend([num_term, term])
            else:
                num_term += " "+term.replace('\\', '/')
                list_of_terms.append(num_term)

        # if one of the patterns not none "number k/m/b" without space
        elif pattern_k is not None or pattern_m is not None or pattern_b is not None:
            list_of_terms.append(term.upper())

        # if the term is not one of the special cases above
        else:
            term_to_list = []
            num_term = term_lower

            # if the previous term was number
            if len(self.term_of_num) > 0:
                num_term = self.transform_number(float(self.term_of_num))
                self.term_of_num = ""
                term_to_list = self.rule_checking(term)  # send the term to recheck rules.

            list_of_terms.extend([num_term]+term_to_list)

        return list_of_terms

    def transform_number(self, num):
        """
        this function will get a number and short it according to the rules.
        :param num: number in float
        :return: shorted number
        """
        num_to_add = ""

        if num < 0.001:
            num = 0.0
        # if the number divide by thousand
        if num/1000 >= 1:
            num = num/1000
            # if the number divide by million
            if num / 1000 >= 1:
                num = num / 1000
                # if the number divide by billion
                if num / 1000 >= 1:
                    num = num / 1000
                    num_as_str = str(num)
                    if "e" in num_as_str:
                        list_of_big_num = num_as_str.split("e+")
                        num_as_str = list_of_big_num[0] + "0"*int(list_of_big_num[1])
                    # round to 3 digits after dot.
                    num_to_add = float(re.match(r'\d+.\d{0,3}', num_as_str).group(0))
                    # check the type of num_to_add
                    if not num_to_add.is_integer():
                        num_to_add = str(num_to_add) + "B"
                    else:
                        num_to_add = str(int(num_to_add)) + "B"

                # divided by million but not billion
                else:
                    num_to_add = float(re.match(r'\d+.\d{0,3}', str(num)).group(0))
                    if not num_to_add.is_integer():
                        num_to_add = str(num_to_add) + "M"
                    else:
                        num_to_add = str(int(num_to_add)) + "M"
            # divided by thousand but not million
            else:
                num_to_add = float(re.match(r'\d+.\d{0,3}', str(num)).group(0))
                if not num_to_add.is_integer():
                    num_to_add = str(num_to_add) + "K"
                else:
                    num_to_add = str(int(num_to_add))+"K"
        # not divided by thousand
        else:
            num_to_add = float(re.match(r'\d+.\d{0,3}', str(num)).group(0))
            if not num_to_add.is_integer():
                num_to_add = str(num_to_add)
            else:
                num_to_add = str(int(num_to_add))

        return num_to_add

    def url_handler(self, pattern, term):
        """
        split urls to terms according to the rules.
        :param pattern: short url cut after domain.
        :param term: full url
        :return: list of terms
        """
        list_of_new_terms = []
        # catch the number of the sub groups
        group_number = len(pattern.groups())

        # loop over each sub group and add it to list_of_new_terms if not none
        for i in range(1, group_number+1):
            curr_group = pattern.group(i)
            if curr_group is not None and i == group_number:
                list_of_new_terms.append(curr_group.lower())
                return list_of_new_terms

        """
        the following code, add to the list all the parts of the url.
        if we want the all parts, uncomment the code and remove the return from above,
        and the after and term.
        """
        # # catch the short url and cut it from the full url.
        # matched_url = pattern.group(0)
        # new_term = term[len(matched_url):]
        #
        # # loop over the rest of the url and add it to list_of_new_terms, ignoring no digit or no letter chars.
        # start_index_curr_term = 0
        # for idx, charr in enumerate(new_term + " "):
        #     if not charr.isalpha() and not charr.isdigit():
        #         if start_index_curr_term == idx:
        #             start_index_curr_term += 1
        #         else:
        #             list_of_new_terms.append(new_term[start_index_curr_term:idx])
        #             start_index_curr_term = idx+1
        #
        # return list_of_new_terms

    def handle_entitie(self):
        """
        this function split the entitie and add each word to dictionary
        :return: list of different words to recognize the entitie
        """
        set_to_return = set()
        lst_entities = self.term_of_entitie.split(" ")

        # loop over the entitie
        for entitie in lst_entities:
            set_of_sub_ent = set()

            # loop over words "name-name"
            for sub_ent in entitie.split("-"):
                if len(sub_ent) > 0:
                    set_of_sub_ent.add(sub_ent.lower())
                    if sub_ent[0].isupper():
                        self.handle_upper_case(sub_ent.lower())

            # if was "name-name" in the entitie
            if len(set_of_sub_ent) != 1:
                set_to_return.update(set_of_sub_ent)

            set_to_return.add(entitie.lower())

        set_to_return.add(self.term_of_entitie.lower())

        # entitie must be more then 1 word.
        # if the entitie doesn't appear already in the current tweet, we will add it to tweet_entities
        if len(set_to_return) > 1 and self.term_of_entitie not in self.tweet_entities:
            self.tweet_entities.append(self.term_of_entitie.lower())

        return set_to_return

    def handle_upper_case(self, term_lower):
        """
        the function adds term_lower to upper case dictionary if not contains the term.
        :param term_lower:
        """
        if term_lower not in self.upper_case_dict:
            self.upper_case_dict[term_lower] = True

    def split_hashtags(self, term):
        """
        the function split the hashtag according to the rules.
        :param term: full hashtag term
        :return: list of new terms splited
        """
        # initalize a set and add to it the term without "_"
        set_of_new_terms = set()
        set_of_new_terms.add(term[1:].replace("_", "").lower())
        # save the term without the '#' sign and initalize vars
        new_term = term[1:]
        term_to_add = ""

        # if the term look like "#word_word_word"
        if '_' in new_term:
            while True:
                # save the index that the sign "_" appear for the first time
                i = new_term.find('_')

                # if we don't find "_" anymore, add to the list and stop the loop
                if i == -1:
                    set_of_new_terms.add(new_term)
                    break

                # cut the term to sub terms. add sub term to list.
                term_to_add = new_term[:i].lower()
                new_term = new_term[i+1:]
                if len(term_to_add) > 0:
                    set_of_new_terms.add(term_to_add)

            # add to set without "_"
            set_of_new_terms.add(term[1:].lower().replace("_", ""))
            set_of_new_terms.add(term.lower().replace("_", ""))
        # if the hashtag look like "#wordWordWord"
        else:
            # loop over the term without "#" and split according to capital letters
            idx = 0
            for letter in term[1:]:
                if letter.isupper() and idx > 0:
                    term_to_add = new_term[:idx].lower()
                    if len(term_to_add) > 1:
                        set_of_new_terms.add(term_to_add)
                    new_term = new_term[idx:]
                    idx = 1
                else:
                    idx += 1

            # add the last term to the set
            set_of_new_terms.add(new_term.lower())

            # catch all the sub terms that separeted by "-" and add to the set
            splited_terms = term[1:].split("-")
            for subterm in splited_terms:
                if len(splited_terms) > 1 and len(subterm) > 0:
                    set_of_new_terms.add(subterm.lower())

            # add to set without "-"
            set_of_new_terms.add(term[1:].lower().replace("-", ""))
            set_of_new_terms.add(term.lower().replace("-", ""))

        return set_of_new_terms

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """
        tweet_id = doc_as_list[0]
        tweet_date = doc_as_list[1]
        # change the date to format "year/month/day hour"
        splited_tweet_date = tweet_date.split(" ")
        tweet_date = splited_tweet_date[5]+"/"+self.month_dict[splited_tweet_date[1]]+"/"+splited_tweet_date[2]+" "+splited_tweet_date[3]

        full_text = doc_as_list[2]
        url = doc_as_list[3]
        retweet_text = doc_as_list[4]
        retweet_url = doc_as_list[5]
        quote_text = doc_as_list[6]
        quote_url = doc_as_list[7]
        term_dict = {}
        tokenized_text = self.parse_sentence(full_text)
        doc_length = len(tokenized_text)  # after text operations.

        for idx, term in enumerate(tokenized_text):
            if term not in term_dict.keys():
                term_dict[term] = 1
            else:
                term_dict[term] += 1

        unique_words = len(term_dict)

        if unique_words == 0:
            tf_max = 0
        else:
            tf_max = max(term_dict.values())

        document = Document(tweet_id, tweet_date, full_text, url, retweet_text, retweet_url, quote_text,
                            quote_url, term_dict, doc_length, tf_max, unique_words)
        return document
