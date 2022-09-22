from pyspark.sql import SparkSession

""" This homework is using Python 3.10 """
""" The only package used in this homework is PySpark. """

if __name__ == '__main__':
    spark = SparkSession.builder.master("local[*]") \
        .appName('My App') \
        .getOrCreate()

    # import two files, select elements that will be used in the later steps, and turn into RDDs.
    meta_rdd = spark.read.json('meta_Musical_Instruments.json')\
        .select('asin', 'price')\
        .rdd
    review_rdd = spark.read.json('reviews_Musical_Instruments.json')\
        .select('asin', 'overall')\
        .rdd

    # step1: set up the key as product ID, and find # of reviews and average ratings for each product ID in review file.
    review_rdd2 = review_rdd.map(lambda x: (x[0], (x[1], 1))) \
        .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
        .mapValues(lambda x: (x[1], x[0] / x[1]))

    # step2: set up the key as product ID, and filter out None values in the meta file.
    meta_rdd2 = meta_rdd.map(lambda x: (x[0], x[1])) \
        .filter(lambda x: x[1] is not None)

    # step3: join two RDDs from previous steps.
    joined_rdd = review_rdd2.join(meta_rdd2)

    # step4: find top 10 products with the greatest number of reviews.
    top_10_list = joined_rdd.takeOrdered(10, key=lambda x: -x[1][0][0])

    # step5: output results to a text file line by line.
    textfile = open("output.txt", "w")
    for line in top_10_list:
        textfile.write('<' + str(line[0]) + '> ')  # product ID
        textfile.write('<' + str(line[1][0][0]) + '> ')  # the number of reviews
        textfile.write('<' + str(line[1][0][1]) + '> ')  # average rating
        textfile.write('<' + str(line[1][1]) + '> ')  # product price
        textfile.write('\n')  # add new line
    textfile.close()

    spark.stop()
