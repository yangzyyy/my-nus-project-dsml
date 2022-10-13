from pyspark import SparkConf, SparkContext
import re
import numpy as np
import nltk

# uncomment the line below to download nltk data package if you haven't already
# nltk.download('punkt')

""" This homework is using Python 3.10 """
""" The packages used in this lab are PySpark, re, NumPy, and nltk. """


def word_tokenize_and_count(content, stopwords):
    """
    Function Name: word_tokenize_and_count
    This function takes in a string, splits it into words, removes
    stopwords, and returns a list of word count.
    Input: content - a string from document content or from a sentence in a document.
           stopwords - a list containing all stopwords.
    Output: tokenized_word_count - a list of word count in the format of [(word1, count), (word2, count), ...].
    """
    tokenized_word_count = sc.parallelize(nltk.word_tokenize(content)) \
        .filter(lambda x: any(c.isalpha() for c in x)) \
        .filter(lambda x: x not in stopwords) \
        .map(lambda x: (x, 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .collect()
    return tokenized_word_count


def term_freq_and_doc_freq(doc_id_to_content_rdd, stopwords):
    """
    Function Name: term_freq_and_doc_freq
    This function counts term frequencies and document frequencies for each word in each document/sentence, and returns
    term frequency as a list and document frequency as a dictionary.
    Input: data_rdd - a pair RDD with keys equal to docID and values equal to context.
           stopwords - a list containing all stopwords.
    Output: term_freq_list - a list of word count, each element is a tuple (doc_id, [(word1, count), (word2, count),..])
            doc_freq_dict - a dictionary of document frequency with keys equal to words and values equal to frequencies.
    """
    term_freq_list = [(doc_id_to_content[0], word_tokenize_and_count(doc_id_to_content[1], stopwords))
                      for doc_id_to_content in doc_id_to_content_rdd.collect()]

    doc_freq_dict = sc.parallelize(term_freq_list) \
        .flatMap(lambda x: x[1]) \
        .groupByKey() \
        .mapValues(len) \
        .collectAsMap()
    return term_freq_list, doc_freq_dict


def normalized_tf_idf(term_freq, doc_freq, n):
    """
    Function Name: normalized_tf_idf
    This function calculates TF-IDF and normalized TF-IDF, and returns normalized TF-IDF as a RDD.
    Input: term_freq - a list of word count of each document/sentence returned from function term_freq_and_doc_freq.
           doc_freq - a dictionary of document frequency returned from function term_freq_and_doc_freq.
           n - number of documents or number of sentences in a document.
    Output: norm_tf_idf_rdd - a RDD with keys equal to docID and values as lists of tuples containing
    words and corresponding normalized TF-IDFs.
    """
    tf_idf_list = sc.parallelize(term_freq) \
        .mapValues(lambda x: [(word, (1 + np.log10(tf)) * np.log10(n / doc_freq[word])) for (word, tf) in list(x)]) \
        .collect()
    sqrt_sum_of_tf_idf_dict = sc.parallelize(tf_idf_list) \
        .mapValues(lambda words_tf_idf: np.sqrt(sum(tf_idf * tf_idf for (word, tf_idf) in list(words_tf_idf)))) \
        .collectAsMap()
    norm_tf_idf_rdd = sc.parallelize(tf_idf_list) \
        .map(lambda x: (x[0], [(word, tf_idf / sqrt_sum_of_tf_idf_dict[x[0]]) for (word, tf_idf) in list(x[1])]))
    return norm_tf_idf_rdd


def relevance(norm_tf_idf_rdd, q, q_norm):
    """
    Function Name: relevance
    This function calculates relevance of each document/sentence w.r.t the query, and return the relevance as a RDD.
    Input: norm_tf_idf_rdd - a RDD of normalized TF-IDFs returned from function normalized_tf_idf.
           q - a RDD containing query vector.
           q_norm - magnitude of the query vector.
    Output: res_rdd - a pair RDD containing relevance of each document/sentence w.r.t query.
    """
    doc_norm_dict = norm_tf_idf_rdd.mapValues(
        lambda x: np.linalg.norm([norm_tf_idf for (word, norm_tf_idf) in list(x)])) \
        .collectAsMap()
    q_words = q.keys().collect()
    rel_rdd = norm_tf_idf_rdd.mapValues(lambda x: np.sum([value
                                                          if word in q_words else 0 for (word, value) in list(x)])) \
        .map(lambda x: (x[0], x[1] / (doc_norm_dict[x[0]] * q_norm)))
    return rel_rdd


def find_top_10(doc_id_to_content_rdd, stopwords, q, q_norm):
    """
    Function Name: find_top_10
    This function goes through step1-step5 in teh Lab2 description and finds top 10 relevant documents.
    Input: doc_id_to_content_rdd - a pair RDD with keys equal to docID and values equal to document contents.
           stopwords - a list containing all stopwords.
           q - a RDD containing query vector.
           q_norm - magnitude of the query vector.
    Output: top_10_res_rdd - a pair RDD with keys equal to docID and values equal to corresponding relevances w.r.t
    provided query.
    """
    n = doc_id_to_content_rdd.count()
    # step 1: compute term frequency of every word in the document.
    tf_list, df_dict = term_freq_and_doc_freq(doc_id_to_content_rdd, stopwords)
    # step 2&3&4: compute TF-IDF, normalized TF-IDF, and relevance in the function relevance.
    norm_rdd = normalized_tf_idf(tf_list, df_dict, n)
    rel_doc_q_rdd = relevance(norm_rdd, q, q_norm)
    # step 5: sort and get top 10 documents
    top_10_list = rel_doc_q_rdd.sortBy(lambda x: x[1], ascending=False).take(10)  # take top 10 relevance as a list
    top_10_res_rdd = sc.parallelize(top_10_list).map(lambda x: (x[0], x[1]))  # store docID and relevance in a RDD
    return top_10_res_rdd


def find_relevant_sentence(id_list, top_10_rdd, stopwords, q, q_norm):
    """
    Function Name: find_relevant_sentence
    This function goes through step6 in Lab2 description and finds most relevant sentence from each document.
    Input: id_list - a list of top 10 document IDs.
           top_10_rdd - a pair RDD with keys equal to docID and values equal to one sentence from the document.
           stopwords - a list containing all stopwords.
           q - a RDD containing query vector.
           q_norm - magnitude of the query vector.
    Output: res_rdd - a pair RDD with keys equal to docID and values as tuples containing relevance and the sentence.
    """
    # create an empty RDD to store most relevant sentence in each document
    res_rdd = sc.parallelize([])
    for doc_id in id_list:
        doc_rdd = top_10_rdd.filter(lambda x: x[0] == doc_id)  # take all sentences from one document
        n_sentence = doc_rdd.count()  # number of sentences in this document

        # calculate TF, TF-IDF, and normalized TF-IDF for every word, and calculate relevance for every sentence
        tf_line_list, df_line_dict = term_freq_and_doc_freq(doc_rdd, stopwords)
        norm_line_rdd = normalized_tf_idf(tf_line_list, df_line_dict, n_sentence)
        # create a new RDD to give each sentence a unique key
        uniq_key_norm_line_list = [(doc_id + '-' + str(i + 1), norm_tf_idf_list)
                                   for i, norm_tf_idf_list in enumerate(norm_line_rdd.values().collect())]
        uniq_key_norm_line_rdd = sc.parallelize(uniq_key_norm_line_list).map(lambda x: (x[0], x[1]))
        rel_line_q_rdd = relevance(uniq_key_norm_line_rdd, q, q_norm)
        # zip each sentence with its corresponding relevance, and convert list to RDD
        rel_sentence_list = list(zip(doc_rdd.values().collect(), rel_line_q_rdd.values().collect()))
        rel_sentence_rdd = sc.parallelize(rel_sentence_list)
        # find the sentence with the maximum relevance
        max_rel_sentence = rel_sentence_rdd.max(key=lambda x: x[1])
        # combine all sentences with maximum relevance from top 10 documents with docID
        res_rdd = res_rdd.union(sc.parallelize([[doc_id, max_rel_sentence]]).map(lambda x: (x[0], x[1])))
    return res_rdd


if __name__ == '__main__':
    # start = time.time()
    conf = SparkConf().setMaster("local").setAppName("My App")
    sc = SparkContext(conf=conf)

    # import datafiles, query, and stopwords as RDDs
    datafile_rdd = sc.wholeTextFiles('datafiles/*')
    datafile_rdd = datafile_rdd.map(lambda x: (re.findall(r'\d+', x[0])[-1], x[1].lower()))

    query_rdd = sc.textFile('query.txt')
    query_words_rdd = query_rdd.flatMap(lambda w: re.split(r'\W+', w))
    query_list = query_words_rdd.collect()
    query_words_rdd = query_words_rdd.map(lambda w: (w, 1))
    query_norm = np.linalg.norm(query_words_rdd.values().collect())  # calculate magnitude of the query vector

    stopwords_rdd = sc.textFile('stopwords.txt')
    stopwords_set = set(stopwords_rdd.collect())

    # STEP1-5: find top 10 relevant documents
    top_10_id_rel_rdd = find_top_10(datafile_rdd, stopwords_set, query_words_rdd, query_norm)

    doc_id_list = top_10_id_rel_rdd.keys().collect()  # docIDs of top 10 documents in a list
    top_10_doc_rdd = datafile_rdd.filter(lambda x: x[0] in doc_id_list)  # filter top 10 documents from original data
    # split sentences by every new line in the document
    top_10_doc_rdd = top_10_doc_rdd.mapValues(lambda x: x.split('\n')).mapValues(lambda x: x[:-1])
    top_10_doc_sentence_rdd = top_10_doc_rdd.flatMap(lambda x: [(x[0], v) for v in x[1]])

    # STEP6: find the most relevant sentence from each document in top 10
    max_rel_sentence_rdd = \
        find_relevant_sentence(doc_id_list, top_10_doc_sentence_rdd, stopwords_set, query_words_rdd, query_norm)

    # combine top 10 documents' ID, document relevance, most relevant sentence, and sentence relevance in one RDD
    combined_res_rdd = top_10_id_rel_rdd.join(max_rel_sentence_rdd).sortBy(lambda x: x[1][0], ascending=False)

    # STEP7: output the result from RDD
    textfile = open("output.txt", "w")
    for line in combined_res_rdd.collect():
        textfile.write('<' + str(line[0]) + '> ')  # docID
        textfile.write('<' + str(line[1][0]) + '> ')  # document relevance score
        textfile.write('<' + str(line[1][1][0]) + '> ')  # relevant sentence
        textfile.write('<' + str(line[1][1][1]) + '> ')  # sentence relevance score
        textfile.write('\n')  # add new line
    textfile.close()
    sc.stop()
