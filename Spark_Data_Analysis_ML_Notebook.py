# Databricks notebook source
import numpy as np


# COMMAND ----------

#part 1
from pyspark.mllib.stat import Statistics

seriesX = sc.parallelize([1.0, 2.0, 3.0, 3.0, 5.0])  # a series
# seriesY must have the same number of partitions and cardinality as seriesX
seriesY = sc.parallelize([11.0, 22.0, 33.0, 33.0, 555.0])

# Compute the correlation using Pearson's method. Enter "spearman" for Spearman's method.
# If a method is not specified, Pearson's method will be used by default.
print("Correlation is: " + str(Statistics.corr(seriesX, seriesY, method="spearman")))

data = sc.parallelize(
    [np.array([1.0, 10.0, 100.0]), np.array([2.0, 20.0, 200.0]), np.array([5.0, 33.0, 366.0])]
)  # an RDD of Vectors

# calculate the correlation matrix using Pearson's method. Use "spearman" for Spearman's method.
# If a method is not specified, Pearson's method will be used by default.
print(Statistics.corr(data, method="spearman"))

# COMMAND ----------

#Part 2
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load the data stored in LIBSVM format as a DataFrame.
sqldf = spark.sql("select * from users_no_labels_3_txt")
data1 = sqldf.rdd.map(list)
print(Statistics.corr(data1, method="pearson"))

# COMMAND ----------

#Part 3

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Load the data stored in LIBSVM format as a DataFrame.

sqldf = spark.sql("select * from colon_cancer_no_labels_4_txt")
data1 = sqldf.rdd.map(list)
print(Statistics.corr(data1, method="pearson"))

# COMMAND ----------

import pandas as pd
ds = sqldf.toPandas()
ds

# COMMAND ----------

#How many data points does this dataset contain?
ds.shape[0]


# COMMAND ----------


#How many attributes does each data point have?
ds.shape[1]

# COMMAND ----------

#How many pairwise similarities should be computed?
ds.shape[1]*(ds.shape[1]-1)/2
