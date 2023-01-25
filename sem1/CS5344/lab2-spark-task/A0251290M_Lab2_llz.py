import string

from pyspark import SparkConf, SparkContext
import re
import numpy as np
import nltk
from collections import Counter
import time

# uncomment the line below to download nltk data package if you haven't already
# nltk.download('punkt')

""" This homework is using Python 3.10 """
""" The packages used in this lab are PySpark, re, NumPy, collections, nltk and nltk_data. """


# TODO add more packages if applicable

def tokenized_word_count(content, stopwords):
    # TODO add documentation
    # given a string content, returns a list of tokenized words with at least one alphabetic character
    # Removes all tokens that do not have any alphabetic characters, e.g. '-', '.', '``' etc.
    # Removes stopwords
    # output: word -> (doc_id, count)
    return sc.parallelize(nltk.word_tokenize(content)) \
        .filter(lambda token: any(c.isalpha() for c in token)) \
        .filter(lambda token: token not in stopwords) \
        .groupBy(lambda x: x) \
        .mapValues(len) \
        .collect()


def compute_word_tf_idf(doc_word_count_dict, num_docs, word_to_num_document_dict):
    # TF - IDF = (1 + log(TF)) * log(N / DF)
    def tf_idf(word, tf):
        return (1 + np.log10(tf)) * np.log10(num_docs / word_to_num_document_dict.get(word))
    return [(word, tf_idf(word, tf)) for (word, tf) in list(doc_word_count_dict)]


def compute_normalised_word_tf_idf(doc_id_and_words_tf_idf, sum_word_tf_idf_per_doc_dict):
    doc_id, words_tf_idf = doc_id_and_words_tf_idf
    doc_sum_word_tf_idf = sum_word_tf_idf_per_doc_dict.get(doc_id)
    word_norm_tf_idfs = [(word, tf_idf / doc_sum_word_tf_idf) for (word, tf_idf) in words_tf_idf]
    return (doc_id, word_norm_tf_idfs)


def term_frequency(doc_id_to_content_rdd, stopwords):
    """
    Function Name: term_frequency
    This function removes stopwords, counts term frequencies for each word in each document/sentence,
    and return two dictionaries.
    Input: doc_id_to_content_rdd - a pair RDD with keys equal to docID and values equal to context
           stopwords - a list containing all stopwords
    """

    start = time.time()
    # TODO see if we can optimise
    per_doc_words_tf = [(doc_id_to_content[0], tokenized_word_count(doc_id_to_content[1], stopwords))
                        for doc_id_to_content in doc_id_to_content_rdd.collect()]

    # dict contains the word -> number of unique documents containing the word
    word_to_num_document_dict = sc.parallelize(per_doc_words_tf) \
        .flatMap(lambda x: x[1]) \
        .groupByKey() \
        .mapValues(len) \
        .collectAsMap()

    num_docs = len(per_doc_words_tf)

    # compute the TF-IDF value of every word w.r.t. each document
    doc_words_tf_idf = sc.parallelize(per_doc_words_tf) \
        .mapValues(lambda x: compute_word_tf_idf(x, num_docs, word_to_num_document_dict)) \
        .collect()

    end = time.time()
    print('cal df and tf:')
    print(end - start)

    start = time.time()

    # compute the normalised TF-IDF value of every word w.r.t. each document
    sum_word_tf_idf_per_doc_dict = sc.parallelize(per_doc_words_tf) \
        .mapValues(lambda words_tf_idf: np.sqrt(sum(tf_idf * tf_idf for (word, tf_idf) in list(words_tf_idf))))\
        .collectAsMap()
    data = sc.parallelize(doc_words_tf_idf) \
        .map(lambda x: compute_normalised_word_tf_idf(x, sum_word_tf_idf_per_doc_dict)) \
        .collectAsMap()

    end = time.time()
    print('norm:')
    print(end - start)

    return data

    # tf_dict = {}
    # uniq_words_rdd = sc.parallelize([])
    # ref = 0  # reference for sentence word count in the same document
    # for (doc_id, content) in doc_id_to_content_rdd.collect():
    #     if doc_id == '1018':
    #         ref += 1
    #         tokenized_words_rdd = word_tokenize(content, stopwords)
    #             # collect unique words from each document/sentence for calculating DF
    #         uniq_words = set(temp_rdd.collect())  # a list containing unique words in current temporary RDD
    #         uniq_words_rdd = uniq_words_rdd.union(sc.parallelize(uniq_words).map(lambda w: (w, 1)))
    #
    #         # compute term frequency
    #         temp_pairs = temp_rdd.map(lambda w: (w, 1))  # make it key-value pair RDD for word count
    #         temp_tf = temp_pairs.reduceByKey(lambda n1, n2: n1 + n2)
    #
    #         if doc_id_to_content_rdd.groupByKey().count() == 1:
    #             tf_dict[doc_id + '-' + str(ref)] = temp_tf  # collect TF for every sentence in the same document
    #         else:
    #             tf_dict[doc_id] = temp_tf  # collect TF for every document
    #
    # df_rdd = uniq_words_rdd.reduceByKey(lambda n1, n2: n1 + n2)  # calculate DF for every word in doc/sentence
    # df_dict = df_rdd.collectAsMap()
    # return tf_dict, df_dict


"""
Function Name: relevance
This function calculates TF-IDF, normalized TF_IDF for each word in each document or sentence,
calculates relevance of each document/sentence w.r.t the query, and return the relevance as a RDD.
Input: tf_dict - a dictionary with keys equal to docID and values including pair RDDs for word count 
                 (output from term-frequency function)
       df_dict - a dictionary with keys equal to docID and values including pair RDDs which contain
                  the count of documents/sentences having the word in current document/sentence 
                  (output from term-frequency function)
       n - total number of documents/sentences
       q - a query RDD
       q_norm - magnitude of the query vector
Output: res_rdd - a pair RDD containing relevance of each document/sentence w.r.t query
"""


def relevance_v2(norm_tf_idf_dict, q, q_norm):
    start = time.time()
    res_rdd = sc.parallelize([])  # create an empty RDD
    for doc_id in norm_tf_idf_dict.keys():
        words_norm_tf_idf = sc.parallelize(norm_tf_idf_dict.get(doc_id))
        # compute relevance of each element (document or sentence) w.r.t the query
        norm = np.linalg.norm(words_norm_tf_idf.values().collect())
        rel = words_norm_tf_idf.join(q).mapValues(lambda x: x[0] * x[1] / (norm * q_norm))
        rel_value = np.sum(rel.values().collect())
        res_rdd = res_rdd.union(sc.parallelize([[doc_id, rel_value]]).map(lambda x: (x[0], x[1])))

    end = time.time()
    print('relevance:')
    print(end - start)
    return res_rdd


def relevance(tf_dict, df_dict, n, q, q_norm):
    res_rdd = sc.parallelize([])  # create an empty RDD
    for key in tf_dict.keys():
        # calculate TF-IDF
        tf_idf_rdd = tf_dict[key].map(lambda x: (x[0], (1 + np.log10(x[1])) * np.log10(n / df_dict[x[0]])))

        # calculate normalized TF-IDF
        sum_of_square_tf_idf = np.sum(np.square(tf_idf_rdd.values().collect()))
        norm_tf_idf_rdd = tf_idf_rdd.mapValues(lambda x: x / np.sqrt(sum_of_square_tf_idf))

        # compute relevance of each element (document or sentence) w.r.t the query
        norm = np.linalg.norm(norm_tf_idf_rdd.values().collect())
        rel = norm_tf_idf_rdd.join(q).mapValues(lambda x: x[0] * x[1] / (norm * q_norm))
        rel_value = np.sum(rel.values().collect())
        res_rdd = res_rdd.union(sc.parallelize([[key, rel_value]]).map(lambda x: (x[0], x[1])))
    return res_rdd


if __name__ == '__main__':

    start = time.time()
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf=conf)

    # import datafiles, query, and stopwords as RDDs
    datafile_rdd = sc.wholeTextFiles('datafiles/*')
    datafile_rdd = datafile_rdd.map(lambda x: (re.findall(r'\d+', x[0])[-1], re.sub('[*]', '', x[1].lower())))

    query_rdd = sc.textFile('query.txt')
    query_words_rdd = query_rdd.flatMap(lambda w: re.split(r'\W+', w))
    query_list = query_words_rdd.collect()
    query_words_rdd = query_words_rdd.map(lambda w: (w, 1))
    query_norm = np.linalg.norm(query_words_rdd.values().collect())  # calculate magnitude of the query vector

    stopwords_rdd = sc.textFile('stopwords.txt')
    stopwords_set = set(stopwords_rdd.collect())

    datafile_words_rdd = datafile_rdd.mapValues(lambda x: x.replace('\n', ' '))  # split words in each document
    # selected_doc_rdd = datafile_words_rdd.filter(lambda x: any(item in x[1] for item in query_list))
    n_doc = datafile_rdd.count()  # n = total number of documents

    end = time.time()
    print('prepare doc:')
    print(end - start)

    # step 1: compute term frequency of every word in the document.
    # tf_doc_dict, df_doc_dict = term_frequency(selected_doc_rdd, stopwords_set)
    norm_tf_idf_dict = term_frequency(datafile_words_rdd, stopwords_set)

    # step 2&3&4: compute TF-IDF, normalized TF-IDF, and relevance in the function relevance.
    # rel_doc_q_rdd = relevance(tf_doc_dict, df_doc_dict, n_doc, query_words_rdd, query_norm)
    rel_doc_q_rdd = relevance_v2(norm_tf_idf_dict, query_words_rdd, query_norm)
    # step 5: sort and get top 10 documents
    top_10_list = rel_doc_q_rdd.sortBy(lambda x: x[1], ascending=False).take(10)  # take top 10 relevance as a list
    top_10_id_rel_rdd = sc.parallelize(top_10_list).map(lambda x: (x[0], x[1]))  # store docID and relevance in a RDD

    doc_id_list = top_10_id_rel_rdd.keys().collect()  # docIDs of top 10 documents in a list
    top_10_doc_rdd = datafile_rdd.filter(lambda x: x[0] in doc_id_list)  # filter top 10 documents from original data
    # split sentences by every new line in the document
    top_10_doc_rdd = top_10_doc_rdd.mapValues(lambda x: x.split('\n')).mapValues(lambda x: x[:-1])
    top_10_doc_sentence_rdd = top_10_doc_rdd.flatMap(lambda x: [(x[0], v) for v in x[1]])

    max_rel_sentence_rdd = sc.parallelize([])  # create an empty RDD
    for doc_id in top_10_doc_rdd.keys().collect():
        doc_rdd = top_10_doc_sentence_rdd.filter(lambda x: x[0] == doc_id)  # take all sentences from one document
        n_sentence = doc_rdd.count()  # number of sentences in this document

        # select sentences that have one or more query words
        selected_sentence_rdd = doc_rdd.filter(lambda x: any(item in x[1] for item in query_list))
        # calculate TF, TF-IDF, and normalized TF-IDF for every word, and calculate relevance for every sentence
        norm_tf_idf_dict = term_frequency(selected_sentence_rdd, stopwords_set)
        # rel_rdd = relevance(tf_line_dict, df_line_dict, n_sentence, query_words_rdd, query_norm)
        rel_rdd = relevance_v2(norm_tf_idf_dict, query_words_rdd, query_norm)

        # zip each sentence with its corresponding relevance, and convert list to RDD
        rel_sentence_list = list(zip(selected_sentence_rdd.values().collect(), rel_rdd.values().collect()))
        rel_sentence_rdd = sc.parallelize(rel_sentence_list)
        # find the sentence with the maximum relevance
        max_rel_sentence = rel_sentence_rdd.max(key=lambda x: x[1])
        # combine all sentences with maximum relevance from top 10 documents with docID
        max_rel_sentence_rdd = max_rel_sentence_rdd.union(sc.parallelize([[doc_id, max_rel_sentence]])
                                                          .map(lambda x: (x[0], x[1])))
    # combine top 10 documents' ID, document relevance, most relevant sentence, and sentence relevance in one RDD
    combined_res_rdd = top_10_id_rel_rdd.join(max_rel_sentence_rdd).sortBy(lambda x: x[1][0], ascending=False)

    # output the result from RDD
    textfile = open("output_llz.txt", "w")
    for line in combined_res_rdd.collect():
        textfile.write('<' + str(line[0]) + '> ')  # docID
        textfile.write('<' + str(line[1][0]) + '> ')  # document relevance score
        textfile.write('<' + str(line[1][1][0]) + '> ')  # relevant sentence
        textfile.write('<' + str(line[1][1][1]) + '> ')  # sentence relevance score
        textfile.write('\n')  # add new line
    textfile.close()
    sc.stop()
