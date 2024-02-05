// Databricks notebook source
// MAGIC %md
// MAGIC
// MAGIC ## Overview
// MAGIC
// MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
// MAGIC
// MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

// COMMAND ----------

// MAGIC %python
// MAGIC # File location and type
// MAGIC file_location = "/FileStore/tables/online_retail_dataset.csv"
// MAGIC file_type = "csv"
// MAGIC
// MAGIC # CSV options
// MAGIC infer_schema = "false"
// MAGIC first_row_is_header = "false"
// MAGIC delimiter = ","
// MAGIC
// MAGIC # The applied options are for CSV files. For other file types, these will be ignored.
// MAGIC df = spark.read.format(file_type) \
// MAGIC   .option("inferSchema", infer_schema) \
// MAGIC   .option("header", first_row_is_header) \
// MAGIC   .option("sep", delimiter) \
// MAGIC   .load(file_location)
// MAGIC
// MAGIC display(df)

// COMMAND ----------

// MAGIC %python
// MAGIC # Create a view or table
// MAGIC
// MAGIC temp_table_name = "online_retail_dataset_csv"
// MAGIC
// MAGIC df.createOrReplaceTempView(temp_table_name)

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC /* Query the created temp table in a SQL cell */
// MAGIC
// MAGIC select * from `online_retail_dataset_csv`

// COMMAND ----------

// MAGIC %python
// MAGIC # With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
// MAGIC # Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
// MAGIC # To do so, choose your table name and uncomment the bottom line.
// MAGIC
// MAGIC permanent_table_name = "online_retail_dataset_csv"
// MAGIC
// MAGIC # df.write.format("parquet").saveAsTable(permanent_table_name)

// COMMAND ----------

val df = spark.read.format("csv")
.option("header", "true")
.option("inferSchema", "true")
.load("/FileStore/tables/online_retail_dataset.csv")
.coalesce(5)
df.cache()
df.createOrReplaceTempView("dfTable")

// COMMAND ----------

// in Scala
import org.apache.spark.sql.functions.count
df.select(count("StockCode")).show() // 541909

// COMMAND ----------

import org.apache.spark.sql.functions.countDistinct
df.select(countDistinct("StockCode")).show() // 4070

// COMMAND ----------

import org.apache.spark.sql.functions.approx_count_distinct
df.select(approx_count_distinct("StockCode", 0.1)).show() // 3364

// COMMAND ----------

// in Scala
import org.apache.spark.sql.functions.{first, last}
df.select(first("StockCode"), last("StockCode")).show()

// COMMAND ----------

// in Scala
import org.apache.spark.sql.functions.{min, max}
df.select(min("Quantity"), max("Quantity")).show()

// COMMAND ----------

// in Scala
import org.apache.spark.sql.functions.sum
df.select(sum("Quantity")).show() // 5176450

// COMMAND ----------

// in Scala
import org.apache.spark.sql.functions.sumDistinct
df.select(sumDistinct("Quantity")).show() // 29310

// COMMAND ----------

import org.apache.spark.sql.functions.{sum, count, avg, expr}
df.select(
count("Quantity").alias("total_transactions"),
sum("Quantity").alias("total_purchases"),
avg("Quantity").alias("avg_purchases"),
expr("mean(Quantity)").alias("mean_purchases"))
.selectExpr(
"total_purchases/total_transactions",
"avg_purchases",
"mean_purchases").show()

// COMMAND ----------


import org.apache.spark.sql.functions.{var_pop, stddev_pop}
import org.apache.spark.sql.functions.{var_samp, stddev_samp}
df.select(var_pop("Quantity"), var_samp("Quantity"),
stddev_pop("Quantity"), stddev_samp("Quantity")).show()

// COMMAND ----------

import org.apache.spark.sql.functions.{skewness, kurtosis}
df.select(skewness("Quantity"), kurtosis("Quantity")).show()


// COMMAND ----------

import org.apache.spark.sql.functions.{corr, covar_pop, covar_samp}
df.select(corr("InvoiceNo", "Quantity"), covar_samp("InvoiceNo", "Quantity"),
covar_pop("InvoiceNo", "Quantity")).show()

// COMMAND ----------

import org.apache.spark.sql.functions.{collect_set, collect_list}
df.agg(collect_set("Country"), collect_list("Country")).show()

// COMMAND ----------

df.groupBy("InvoiceNo", "CustomerId").count().show()

// COMMAND ----------

import org.apache.spark.sql.functions.count
df.groupBy("InvoiceNo").agg(
count("Quantity").alias("quan"),
expr("count(Quantity)")).show()

// COMMAND ----------

df.groupBy("InvoiceNo").agg("Quantity"->"avg", "Quantity"->"stddev_pop").show()

// COMMAND ----------

import org.apache.spark.sql.functions.{col, to_date}
val dfWithDate = df.withColumn("date", to_date(col("InvoiceDate"),
"MM/d/yyyy H:mm"))
dfWithDate.createOrReplaceTempView("dfWithDate")

// COMMAND ----------

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.col
val windowSpec = Window
.partitionBy("CustomerId", "date")
.orderBy(col("Quantity").desc)
.rowsBetween(Window.unboundedPreceding, Window.currentRow)

// COMMAND ----------

import org.apache.spark.sql.functions.max
val maxPurchaseQuantity = max(col("Quantity")).over(windowSpec)


// COMMAND ----------

import org.apache.spark.sql.functions.{dense_rank, rank}
val purchaseDenseRank = dense_rank().over(windowSpec)
val purchaseRank = rank().over(windowSpec)

// COMMAND ----------

import org.apache.spark.sql.functions.col
spark.conf.set("spark.sql.legacy.timeParserPolicy", "LEGACY")
dfWithDate.where("CustomerId IS NOT NULL").orderBy("CustomerId")
.select(
col("CustomerId"),
col("date"),
col("Quantity"),
purchaseRank.alias("quantityRank"),
purchaseDenseRank.alias("quantityDenseRank"),
maxPurchaseQuantity.alias("maxPurchaseQuantity")).show()

// COMMAND ----------

val dfNoNull = dfWithDate.drop()
dfNoNull.createOrReplaceTempView("dfNoNull")

// COMMAND ----------

val rolledUpDF = dfNoNull.rollup("Date", "Country").agg(sum("Quantity"))
.selectExpr("Date", "Country", "`sum(Quantity)` as total_quantity")
.orderBy("Date")
rolledUpDF.show()

// COMMAND ----------

rolledUpDF.where("Country IS NULL").show()
rolledUpDF.where("Date IS NULL").show()

// COMMAND ----------

dfNoNull.cube("Date", "Country").agg(sum(col("Quantity")))
.select("Date", "Country", "sum(Quantity)").orderBy("Date").show()

// COMMAND ----------

import org.apache.spark.sql.functions.{grouping_id, sum, expr}
dfNoNull.cube("customerId", "stockCode").agg(grouping_id(), sum("Quantity"))
.orderBy(expr("grouping_id()").desc)
.show()

// COMMAND ----------

val pivoted = dfWithDate.groupBy("date").pivot("Country").sum()


// COMMAND ----------

import org.apache.spark.sql.expressions.MutableAggregationBuffer
import org.apache.spark.sql.expressions.UserDefinedAggregateFunction
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
class BoolAnd extends UserDefinedAggregateFunction {
def inputSchema: org.apache.spark.sql.types.StructType =
StructType(StructField("value", BooleanType) :: Nil)
def bufferSchema: StructType = StructType(
StructField("result", BooleanType) :: Nil
)
def dataType: DataType = BooleanType
def deterministic: Boolean = true
def initialize(buffer: MutableAggregationBuffer): Unit = {
buffer(0) = true
}
def update(buffer: MutableAggregationBuffer, input: Row): Unit = {
buffer(0) = buffer.getAs[Boolean](0) && input.getAs[Boolean](0)
}
def merge(buffer1: MutableAggregationBuffer, buffer2: Row): Unit = {
buffer1(0) = buffer1.getAs[Boolean](0) && buffer2.getAs[Boolean](0)
}
def evaluate(buffer: Row): Any = {
buffer(0)
}
}

// COMMAND ----------


val ba = new BoolAnd
spark.udf.register("booland", ba)
import org.apache.spark.sql.functions._
spark.range(1)
.selectExpr("explode(array(TRUE, TRUE, TRUE)) as t")
.selectExpr("explode(array(TRUE, FALSE, TRUE)) as f", "t")
.select(ba(col("t")), expr("booland(f)"))
.show()

// COMMAND ----------


