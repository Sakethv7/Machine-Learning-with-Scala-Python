// Databricks notebook source
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{SQLContext, SparkSession}
import io.delta.tables._
import org.apache.spark.sql.functions._
import org.apache.commons.io.FileUtils
import java.io.File
 
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{SQLImplicits, SparkSession}
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.MutableAggregationBuffer
 
import scala.reflect.internal.util.FileUtils
 
Logger.getLogger("org").setLevel(Level.OFF)
println(s"Delta Test ${new java.util.Date()} ") // seamless integration with Java
type S = String;
type I = Integer;
type D = Double;
type B = Boolean // type synonyms
val spark = SparkSession
.builder()
.appName("DeltaTest ")
.master("local[*]")
.config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
.config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
.getOrCreate()
spark.conf.set("spark.sql.shuffle.partitions", "5")
import spark.implicits._
println(s"Set up a local file for a parquet dataset, /Users/rob/tmp/delta-table")
// the FileUtils was ambiguous so I used the complete path
val file = new File("/Users/rob/tmp/delta-table")
if (file.exists()) org.apache.commons.io.FileUtils.deleteDirectory(file)
 
println("get the (canonical) file path and then Create a table from spark.range(0,5)")
val path = file.getCanonicalPath
println(s" Path to new file.getCanonicalPath  is $path")
println(s"  'VAR' write out a range(0,5).toDF(nr) to the path as a delta file ,I made the col = nr  ")
var data = spark.range(0, 5).toDF("nr")
data.write.format("delta").save(path)
println(path)
 
println(s" Now read that df=range(0,5).toDF(nr) parquet file back in and show it")
val df = spark.read.format("delta").load(path)
df.show()
println(s" print the schema for the range(0,5).toDF")
df.printSchema()

// COMMAND ----------

println(s" New operation 'Upsert (merge)' new data  ")
println("Upsert 'newData', spark.range(0, 20).toDF(nr) with oldData range(0,5).toDF(nr)")
println(s"alias the old table as 'oldData' , alias the new table as 'newData' ")
val newData = spark.range(0, 20).toDF("nr")
val deltaTable = DeltaTable.forPath(path)
deltaTable.as("oldData")
.merge(
newData.as("newData"),
"oldData.nr = newData.nr")
.whenMatched
.update(Map("nr" -> col("newData.nr")))
.whenNotMatched
.insert(Map("nr" -> col("newData.nr")))
.execute()
println(s" Now show the merged table")
deltaTable.toDF.show()

// COMMAND ----------

println(s" Now Overwrite the current table, with data = spark.range(5, 10 ), to path $path ")
// Update table data, why no 'val' here ?
data = spark.range(5, 10).toDF("nr")
data.write.format("delta").mode("overwrite").save(path)
 
println(s" Now show overwritten table ")
deltaTable.toDF.show()

// COMMAND ----------

// Update every even value by adding 100 to it
println("Update to the table (add 100 to every even value)")
deltaTable.update(
condition = expr("nr % 2 == 0"),
set = Map("nr" -> expr("nr + 100")))
deltaTable.toDF.show()

// COMMAND ----------

println(s" Now delete every even value and show")
// Delete every even value
deltaTable.delete(condition = expr("nr % 2 == 0"))
deltaTable.toDF.show()

// COMMAND ----------

println(s" Now read old data using 'time travel' ")
// Read old version of the data using time travel
print("Read old data using time travel   version 0?")
val df2 = spark.read.format("delta")
          .option("versionAsOf", 0).load(path)
df2.show()

// COMMAND ----------

println(s" if you want to see what was written, don't deleteDirectory  ")
println(s" using FileUtils, delete the directory ")
org.apache.commons.io.FileUtils.deleteDirectory(file)
println(s" spark.stop() ")
spark.stop()

// COMMAND ----------


