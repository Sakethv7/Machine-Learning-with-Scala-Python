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
// MAGIC file_location = "/FileStore/tables/2015_summary.json"
// MAGIC file_type = "json"
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
// MAGIC temp_table_name = "2015_summary_json"
// MAGIC
// MAGIC df.createOrReplaceTempView(temp_table_name)

// COMMAND ----------

// MAGIC %sql
// MAGIC
// MAGIC /* Query the created temp table in a SQL cell */
// MAGIC
// MAGIC select * from `2015_summary_json`

// COMMAND ----------

// MAGIC %python
// MAGIC # With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
// MAGIC # Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
// MAGIC # To do so, choose your table name and uncomment the bottom line.
// MAGIC
// MAGIC permanent_table_name = "2015_summary_json"
// MAGIC
// MAGIC # df.write.format("parquet").saveAsTable(permanent_table_name)

// COMMAND ----------

val df = spark.read.format("json")
.load("/FileStore/tables/2015_summary.json")

// COMMAND ----------

spark.read.format("json").load("/FileStore/tables/2015_summary.json").schema

// COMMAND ----------

import org.apache.spark.sql.types.{StructField, StructType, StringType, LongType}
import org.apache.spark.sql.types.Metadata

// COMMAND ----------

val myManualSchema = StructType(Array(
StructField("DEST_COUNTRY_NAME", StringType, true),
StructField("ORIGIN_COUNTRY_NAME", StringType, true),
StructField("count", LongType, false,
Metadata.fromJson("{\"hello\":\"world\"}"))
))
val df = spark.read.format("json").schema(myManualSchema)
.load("/FileStore/tables/2015_summary.json")

// COMMAND ----------

import org.apache.spark.sql.functions.{col, column}
col("someColumnName")
column("someColumnName")

// COMMAND ----------

import org.apache.spark.sql.functions.expr
expr("(((someCol + 5) * 200) - 6) < otherCol")

// COMMAND ----------

import org.apache.spark.sql.Row
val myRow = Row("Hello", null, 1, false)

// COMMAND ----------

myRow(0) // type Any
myRow(0).asInstanceOf[String] // String
myRow.getString(0) // String
myRow.getInt(2) // Int

// COMMAND ----------

val df = spark.read.format("json")
.load("/FileStore/tables/2015_summary.json")
df.createOrReplaceTempView("dfTable")

// COMMAND ----------

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{StructField, StructType, StringType, LongType}

// COMMAND ----------

val myDF = Seq(("Hello", 2, 1L)).toDF("col1", "col2", "col3")
myDF.show()

// COMMAND ----------

df.select("DEST_COUNTRY_NAME").show(2)

// COMMAND ----------

df.select("DEST_COUNTRY_NAME", "ORIGIN_COUNTRY_NAME").show()

// COMMAND ----------

import org.apache.spark.sql.functions.{expr, col, column}
df.select(
df.col("DEST_COUNTRY_NAME"),
col("DEST_COUNTRY_NAME"),
column("DEST_COUNTRY_NAME"),
'DEST_COUNTRY_NAME,
$"DEST_COUNTRY_NAME",
expr("DEST_COUNTRY_NAME"))
.show(2)

// COMMAND ----------

df.select(expr("DEST_COUNTRY_NAME AS destination")).show(2)

// COMMAND ----------

df.select(expr("DEST_COUNTRY_NAME as destination").alias("DEST_COUNTRY_NAME"))
.show(2)

// COMMAND ----------

df.selectExpr("DEST_COUNTRY_NAME as newColumnName", "DEST_COUNTRY_NAME").show(2)


// COMMAND ----------

df.selectExpr(
"*", // include all original columns
"(DEST_COUNTRY_NAME = ORIGIN_COUNTRY_NAME) as withinCountry")
.show(2)

// COMMAND ----------

df.selectExpr("avg(count)", "count(distinct(DEST_COUNTRY_NAME))").show(2)

// COMMAND ----------

import org.apache.spark.sql.functions.lit
df.select(expr("*"), lit(1).as("One")).show(2)

// COMMAND ----------

df.withColumn("numberOne", lit(1)).show(2)

// COMMAND ----------

df.withColumn("withinCountry", expr("ORIGIN_COUNTRY_NAME == DEST_COUNTRY_NAME"))
.show(2)

// COMMAND ----------

df.withColumnRenamed("DEST_COUNTRY_NAME", "dest").columns

// COMMAND ----------

import org.apache.spark.sql.functions.expr
import org.apache.spark.sql.functions.expr
val dfWithLongColName = df.withColumn(
"This Long Column-Name",
expr("ORIGIN_COUNTRY_NAME"))

// COMMAND ----------

dfWithLongColName.selectExpr(
"`This Long Column-Name`",
"`This Long Column-Name` as `new col`")
.show(2)

// COMMAND ----------

dfWithLongColName.select(col("This Long Column-Name")).columns

// COMMAND ----------

df.filter(col("count") < 2).show(2)


// COMMAND ----------

df.where(col("count") < 2).where(col("ORIGIN_COUNTRY_NAME") =!= "Croatia")
.show(2)

// COMMAND ----------

df.select("ORIGIN_COUNTRY_NAME", "DEST_COUNTRY_NAME").distinct().count()

// COMMAND ----------

df.select("ORIGIN_COUNTRY_NAME").distinct().count()

// COMMAND ----------

val seed = 5
val withReplacement = false
val fraction = 0.5
df.sample(withReplacement, fraction, seed).count()

val dataFrames = df.randomSplit(Array(0.25, 0.75), seed)
dataFrames(0).count() > dataFrames(1).count() // False

// COMMAND ----------

import org.apache.spark.sql.Row
val schema = df.schema
val newRows = Seq(
Row("New Country", "Other Country", 5L),
Row("New Country 2", "Other Country 3", 1L))
val parallelizedRows = spark.sparkContext.parallelize(newRows)
val newDF = spark.createDataFrame(parallelizedRows, schema)
df.union(newDF)
.where("count = 1")
.where($"ORIGIN_COUNTRY_NAME" =!= "United States")
.show() // get all of them and we'll see our new rows at the end

// COMMAND ----------

df.sort("count").show(5)
df.orderBy("count", "DEST_COUNTRY_NAME").show(5)
df.orderBy(col("count"), col("DEST_COUNTRY_NAME")).show(5)

// COMMAND ----------

import org.apache.spark.sql.functions.{desc, asc}
df.orderBy(expr("count desc")).show(2)
df.orderBy(desc("count"), asc("DEST_COUNTRY_NAME")).show(2)

// COMMAND ----------

spark.read.format("json").load("/FileStore/tables/2015_summary.json")
.sortWithinPartitions("count")

// COMMAND ----------

df.limit(5).show()

// COMMAND ----------

df.orderBy(expr("count desc")).limit(6).show()

// COMMAND ----------

df.rdd.getNumPartitions // 1
df.repartition(5)

// COMMAND ----------

df.repartition(col("DEST_COUNTRY_NAME"))

// COMMAND ----------

df.repartition(5, col("DEST_COUNTRY_NAME"))

// COMMAND ----------

df.repartition(5, col("DEST_COUNTRY_NAME")).coalesce(2)

// COMMAND ----------

val collectDF = df.limit(10)
collectDF.take(5) // take works with an Integer count
collectDF.show() // this prints it out nicely

// COMMAND ----------

collectDF.show(5, false)
collectDF.collect()

// COMMAND ----------


