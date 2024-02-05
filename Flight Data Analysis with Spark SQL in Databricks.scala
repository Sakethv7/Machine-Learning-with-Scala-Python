// Databricks notebook source
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.Row


// Task 1: Read in the Flight data
val spark1 = SparkSession.builder().appName("ADVBD2023").master("Local[*]").getOrCreate()
spark.conf.set ("spark.sql.shuffle.partitions","5")
val path1 = "/FileStore/tables/2015_summary-2.csv"
val my_schema=StructType(Array (StructField("DEST_COUNTRY_NAME",StringType, false),StructField("ORIGIN_COUNTRY_NAME",StringType, false), StructField("count",LongType,false)))
val df = spark.read.option ("header", "true").format ("csv").schema (my_schema).load (path1)
df.show(5)


// COMMAND ----------

val df_Renamed = df.withColumnRenamed("DEST_COUNTRY_NAME","Destinations").withColumnRenamed("ORIGIN_COUNTRY_NAME","Origins")
df_Renamed.columns
df_Renamed.show(5)


// COMMAND ----------

val sort_df = df_Renamed.sort(desc("count"))
sort_df.show(10)


// COMMAND ----------

sort_df.createOrReplaceTempView("flight_table")
val sqlview = spark.sql("SELECT Destinations, Origins, count FROM flight_table")
sqlview.show(10)


// COMMAND ----------


