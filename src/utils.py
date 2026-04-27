import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import SparkSession


def create_spark():
    """
    Create Spark session.
    """
    return SparkSession.builder.appName("KMeansManual").getOrCreate()


def load_data(spark, path):
    """
    Load CSV and return RDD + labels.
    """

    df = spark.read.csv(path, header=True, inferSchema=True)

    # Extract labels
    class_labels = df.select('class').rdd.map(lambda x: x[0]).collect()

    # Remove label column
    df = df[df.columns[:-1]]

    rdd = df.rdd.map(lambda row: np.array(row))

    return rdd, class_labels


def normalize_rdd(spark, rdd):
    """
    Normalize RDD using MinMax scaling.
    """

    data = np.array(rdd.collect())
    scaled = MinMaxScaler().fit_transform(data)

    return spark.sparkContext.parallelize(scaled)
