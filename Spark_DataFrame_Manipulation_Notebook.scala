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
// MAGIC file_location = "/FileStore/tables/2010_12_01-1.csv"
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
// MAGIC temp_table_name = "csv_20101201_1"
// MAGIC df.createOrReplaceTempView(temp_table_name)
// MAGIC
// MAGIC
// MAGIC

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC /* Query the created temp table in a SQL cell */
// MAGIC
// MAGIC select * from `csv_20101201_1`

// COMMAND ----------

// MAGIC %python
// MAGIC # With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
// MAGIC # Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
// MAGIC # To do so, choose your table name and uncomment the bottom line.
// MAGIC
// MAGIC permanent_table_name = "2010_12_01-1_csv"
// MAGIC
// MAGIC # df.write.format("parquet").saveAsTable(permanent_table_name)

// COMMAND ----------

val df = spark.read.format("csv")
.option("header", "true")
.option("inferSchema", "true")
.load("/FileStore/tables/2010_12_01.csv")
df.printSchema()
df.createOrReplaceTempView("dfTable")

// COMMAND ----------

import org.apache.spark.sql.functions.col
df.where(col("InvoiceNo").equalTo(536365))
.select("InvoiceNo", "Description")
.show(5, false)

// COMMAND ----------

import org.apache.spark.sql.functions.col
df.where(col("InvoiceNo") === 536365)
.select("InvoiceNo", "Description")
.show(5, false)

// COMMAND ----------

val priceFilter = col("UnitPrice") > 600
val descripFilter = col("Description").contains("POSTAGE")
df.where(col("StockCode").isin("DOT")).where(priceFilter.or(descripFilter))
.show()

// COMMAND ----------

val DOTCodeFilter = col("StockCode") === "DOT"
val priceFilter = col("UnitPrice") > 600
val descripFilter = col("Description").contains("POSTAGE")
df.withColumn("isExpensive", DOTCodeFilter.and(priceFilter.or(descripFilter)))
.where("isExpensive")
.select("unitPrice", "isExpensive").show(5)

// COMMAND ----------

import org.apache.spark.sql.functions.{expr, not, col}
df.withColumn("isExpensive", not(col("UnitPrice").leq(250)))
.filter("isExpensive")
.select("Description", "UnitPrice").show(5)

// COMMAND ----------

import org.apache.spark.sql.functions.{corr}
df.stat.corr("Quantity", "UnitPrice")
df.select(corr("Quantity", "UnitPrice")).show()

// COMMAND ----------


import org.apache.spark.sql.functions.regexp_replace
val simpleColors = Seq("black", "white", "red", "green", "blue")
val regexString = simpleColors.map(_.toUpperCase).mkString("|")
// the | signifies `OR` in regular expression syntax
df.select(
regexp_replace(col("Description"), regexString, "COLOR").alias("color_clean"),
col("Description")).show(2)

// COMMAND ----------

import org.apache.spark.sql.functions.translate
  df.select(translate(col("Description"), "LEET", "1337"), col("Description"))
.show(2)	



// COMMAND ----------

import org.apache.spark.sql.functions.regexp_extract
  val regexString = simpleColors.map(_.toUpperCase).mkString("(", "|", ")")
  df.select(
       regexp_extract(col("Description"), regexString, 1).alias("color_clean"),
       col("Description")).show(2)
       


// COMMAND ----------

val containsBlack = col("Description").contains("BLACK")
  val containsWhite = col("DESCRIPTION").contains("WHITE")
  df.withColumn("hasSimpleColor", containsBlack.or(containsWhite))
    .where("hasSimpleColor")
    .select("Description").show(3, false)


// COMMAND ----------

spark.sql("""
  SELECT
ifnull(null, 'return_value'),
nullif('value', 'value'),
nvl(null, 'return_value'),
nvl2('not_null', 'return_value', "else_value")
FROM dfTable LIMIT 1
  """)
  .take(5)
  .foreach(println)


// COMMAND ----------


import org.apache.spark.sql.functions.map
df.select(map(col("Description"), col("InvoiceNo")).alias("complex_map")).show(2)

// COMMAND ----------

df.select(map(col("Description"), col("InvoiceNo")).alias("complex_map"))
.selectExpr("complex_map['WHITE METAL LANTERN']").show(2)

// COMMAND ----------

// Map
df.select(map(col("Description"), col("InvoiceNo")).alias("complex_map"))
.selectExpr("explode(complex_map)").show(2)

// COMMAND ----------

//Struct
import org.apache.spark.sql.functions.struct
val complexDF = df.select(struct("Description", "InvoiceNo").alias("complex"))
complexDF.createOrReplaceTempView("complexDF")

// COMMAND ----------

//Arrays
import org.apache.spark.sql.functions.split
df.select(split(col("Description"), " ")).show(2)

// COMMAND ----------

df.select(split(col("Description"), " ").alias("array_col"))
.selectExpr("array_col[0]").show(2)

// COMMAND ----------

import org.apache.spark.sql.functions.size
df.select(size(split(col("Description"), " "))).show(2)

// COMMAND ----------

import org.apache.spark.sql.functions.array_contains
df.select(array_contains(split(col("Description"), " "), "WHITE")).show(2)

// COMMAND ----------

import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.SparkSession


val spark = SparkSession.builder()
  .appName("UDFExample")
  .getOrCreate()


val udfExampleDF = spark.range(5).toDF("num")


val power3 = (number: Double) => number * number * number


spark.udf.register("power3", power3)


val resultDF = udfExampleDF.selectExpr("power3(num) as cubed")


resultDF.show()


// COMMAND ----------


