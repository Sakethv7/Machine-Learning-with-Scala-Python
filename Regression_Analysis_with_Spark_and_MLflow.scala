// Databricks notebook source
val filePath ="/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/"
val airbnbDF = spark.read.parquet(filePath)
airbnbDF.select("neighbourhood_cleansed", "room_type", "bedrooms", "bathrooms",
 "number_of_reviews", "price").show(5)


// COMMAND ----------

val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42) 
println(f"""There are ${trainDF.count} rows in the training set, and ${testDF.count} in the test set""")


// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
val vecAssembler = new VectorAssembler()
 .setInputCols(Array("bedrooms"))
 .setOutputCol("features")
val vecTrainDF = vecAssembler.transform(trainDF)
vecTrainDF.select("bedrooms", "features", "price").show(10)


// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression
val lr = new LinearRegression()
 .setFeaturesCol("features")
.setLabelCol("price")
val lrModel = lr.fit(vecTrainDF)


// COMMAND ----------

val m = lrModel.coefficients(0) 
val b = lrModel.intercept 
println(f"""The formula for the linear regression line is price = $m%1.2f*bedrooms + $b%1.2f""")


// COMMAND ----------

import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(vecAssembler, lr))
val pipelineModel = pipeline.fit(trainDF)


// COMMAND ----------

val predDF = pipelineModel.transform(testDF)
predDF.select("bedrooms", "features", "price", "prediction").show(10)


// COMMAND ----------

import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer}
val categoricalCols = trainDF.dtypes.filter(_._2 == "StringType").map(_._1)
val indexOutputCols = categoricalCols.map(_ + "Index")
val oheOutputCols = categoricalCols.map(_ + "OHE")
val stringIndexer = new StringIndexer()
 .setInputCols(categoricalCols)
 .setOutputCols(indexOutputCols)
 .setHandleInvalid("skip")
val oheEncoder = new OneHotEncoder()
 .setInputCols(indexOutputCols)
 .setOutputCols(oheOutputCols)
val numericCols = trainDF.dtypes.filter{ case (field, dataType) =>
 dataType == "DoubleType" && field != "price"}.map(_._1)
val assemblerInputs = oheOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
 .setInputCols(assemblerInputs)
 .setOutputCol("features")


// COMMAND ----------

import org.apache.spark.ml.feature.RFormula
val rFormula = new RFormula()
 .setFormula("price ~ .")
 .setFeaturesCol("features")
 .setLabelCol("price")
 .setHandleInvalid("skip")

// COMMAND ----------

import org.apache.spark.ml.feature.RFormula
val rFormula = new RFormula()
 .setFormula("price ~ .")
 .setFeaturesCol("features")
 .setLabelCol("price")
 .setHandleInvalid("skip")

// COMMAND ----------

val lr = new LinearRegression()
 .setLabelCol("price")
 .setFeaturesCol("features")
val pipeline = new Pipeline()
 .setStages(Array(stringIndexer, oheEncoder, vecAssembler, lr))
// Or use RFormula
// val pipeline = new Pipeline().setStages(Array(rFormula, lr))
val pipelineModel = pipeline.fit(trainDF)
val predDF = pipelineModel.transform(testDF)
predDF.select("features", "price", "prediction").show(5)

// COMMAND ----------

import org.apache.spark.ml.evaluation.RegressionEvaluator
val regressionEvaluator = new RegressionEvaluator()
 .setPredictionCol("prediction")
 .setLabelCol("price")
 .setMetricName("rmse")
val rmse = regressionEvaluator.evaluate(predDF)
println(f"RMSE is $rmse%.1f")


// COMMAND ----------

val r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
println(s"R2 is $r2")

// COMMAND ----------

val pipelinePath = "/tmp/lr-pipeline-model" 
pipelineModel.write.overwrite().save(pipelinePath)

// COMMAND ----------

import org.apache.spark.ml.PipelineModel
val savedPipelineModel = PipelineModel.load(pipelinePath)

import org.apache.spark.ml.regression.DecisionTreeRegressor
val dt = new DecisionTreeRegressor()
 .setLabelCol("price")
// Filter for just numeric columns (and exclude price, our label)
val numericCols = trainDF.dtypes.filter{ case (field, dataType) =>
 dataType == "DoubleType" && field != "price"}.map(_._1)
// Combine output of StringIndexer defined above and numeric columns
val assemblerInputs = indexOutputCols ++ numericCols
val vecAssembler = new VectorAssembler()
 .setInputCols(assemblerInputs)
 .setOutputCol("features")
// Combine stages into pipeline
val stages = Array(stringIndexer, vecAssembler, dt)
val pipeline = new Pipeline()
 .setStages(stages)
val pipelineModel = pipeline.fit(trainDF)


// COMMAND ----------

dt.setMaxBins(40)
val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

import org.apache.spark.ml.regression.RandomForestRegressor
val rf = new RandomForestRegressor()
 .setLabelCol("price")
 .setMaxBins(40)
 .setSeed(42)

// COMMAND ----------

val pipeline = new Pipeline()
 .setStages(Array(stringIndexer, vecAssembler, rf))

// COMMAND ----------

import org.apache.spark.ml.tuning.ParamGridBuilder
val paramGrid = new ParamGridBuilder()
 .addGrid(rf.maxDepth, Array(2, 4, 6))
 .addGrid(rf.numTrees, Array(10, 100))
 .build()



// COMMAND ----------

val evaluator = new RegressionEvaluator()
 .setLabelCol("price")
 .setPredictionCol("prediction")
 .setMetricName("rmse")



// COMMAND ----------

 import org.apache.spark.ml.tuning.CrossValidator
 val cv = new CrossValidator()
.setEstimator(pipeline)
.setEvaluator(evaluator)
.setEstimatorParamMaps(paramGrid)
.setNumFolds(3)
.setSeed(42)
val cvModel = cv.fit(trainDF)

// COMMAND ----------

cvModel.getEstimatorParamMaps.zip(cvModel.avgMetrics)

// COMMAND ----------

val cvModel = cv.setParallelism(4).fit(trainDF)


// COMMAND ----------

val cv = new CrossValidator()
.setEstimator(rf) 
.setEvaluator(evaluator) 
.setEstimatorParamMaps(paramGrid) 
.setNumFolds(3) 
.setParallelism(4) 
.setSeed(42) 
val pipeline = new Pipeline() 
.setStages(Array(stringIndexer, vecAssembler, cv)) 
val pipelineModel = pipeline.fit(trainDF)

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.ml import Pipeline
// MAGIC from pyspark.ml.feature import StringIndexer, VectorAssembler
// MAGIC from pyspark.ml.regression import RandomForestRegressor
// MAGIC from pyspark.ml.evaluation import RegressionEvaluator
// MAGIC
// MAGIC filePath = """/databricks-datasets/learning-spark-v2/sf-airbnb/sf-airbnb-clean.parquet/"""
// MAGIC airbnbDF = spark.read.parquet(filePath)
// MAGIC (trainDF, testDF) = airbnbDF.randomSplit([.8, .2], seed=42)
// MAGIC categoricalCols = [field for (field, dataType) in trainDF.dtypes
// MAGIC  if dataType == "string"]
// MAGIC indexOutputCols = [x + "Index" for x in categoricalCols]
// MAGIC stringIndexer = StringIndexer(inputCols=categoricalCols,
// MAGIC  outputCols=indexOutputCols,
// MAGIC  handleInvalid="skip")
// MAGIC numericCols = [field for (field, dataType) in trainDF.dtypes
// MAGIC  if ((dataType == "double") & (field != "price"))]
// MAGIC assemblerInputs = indexOutputCols + numericCols
// MAGIC vecAssembler = VectorAssembler(inputCols=assemblerInputs,
// MAGIC  outputCol="features")
// MAGIC rf = RandomForestRegressor(labelCol="price", maxBins=40, maxDepth=5,
// MAGIC  numTrees=100, seed=42)
// MAGIC pipeline = Pipeline(stages=[stringIndexer, vecAssembler, rf])
// MAGIC

// COMMAND ----------

// MAGIC %python
// MAGIC import mlflow
// MAGIC import mlflow.spark
// MAGIC import pandas as pd
// MAGIC with mlflow.start_run(run_name="random-forest") as run:
// MAGIC  # Log params: num_trees and max_depth
// MAGIC  mlflow.log_param("num_trees", rf.getNumTrees())
// MAGIC  mlflow.log_param("max_depth", rf.getMaxDepth())
// MAGIC  # Log model
// MAGIC  pipelineModel = pipeline.fit(trainDF)
// MAGIC  mlflow.spark.log_model(pipelineModel, "model")
// MAGIC  # Log metrics: RMSE and R2
// MAGIC  predDF = pipelineModel.transform(testDF)
// MAGIC  regressionEvaluator = RegressionEvaluator(predictionCol="prediction",
// MAGIC  labelCol="price")
// MAGIC  rmse = regressionEvaluator.setMetricName("rmse").evaluate(predDF)
// MAGIC  r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
// MAGIC  mlflow.log_metrics({"rmse": rmse, "r2": r2})
// MAGIC  # Log artifact: feature importance scores
// MAGIC  rfModel = pipelineModel.stages[-1]
// MAGIC  pandasDF = (pd.DataFrame(list(zip(vecAssembler.getInputCols(),
// MAGIC  rfModel.featureImportances)),
// MAGIC  columns=["feature", "importance"])
// MAGIC  .sort_values(by="importance", ascending=False))
// MAGIC  # First write to local filesystem, then tell MLflow where to find that file
// MAGIC  pandasDF.to_csv("feature-importance.csv", index=False)
// MAGIC  mlflow.log_artifact("feature-importance.csv")
// MAGIC
// MAGIC

// COMMAND ----------

// MAGIC %python
// MAGIC from mlflow.tracking import MlflowClient
// MAGIC client = MlflowClient()
// MAGIC runs = client.search_runs(run.info.experiment_id,order_by=["attributes.start_time desc"],max_results=1)
// MAGIC run_id = runs[0].info.run_id
// MAGIC runs[0].data.metrics
// MAGIC

// COMMAND ----------

import org.mlflow.tracking.ActiveRun
import org.mlflow.tracking.MlflowContext
import java.io.{File,PrintWriter}

// COMMAND ----------

val mlflowContext = new MlflowContext()

val experimentName = "/Shared/Quickstart"
val client = mlflowContext.getClient()
val experimentOpt = client.getExperimentByName(experimentName);
if (!experimentOpt.isPresent()) {
  client.createExperiment(experimentName)
}
mlflowContext.setExperimentName(experimentName)


// COMMAND ----------

import java.nio.file.Paths
val run = mlflowContext.startRun("run")
// Log a parameter (key-value pair)
run.logParam("param1", "5")
 
// Log a metric; metrics can be updated throughout the run
run.logMetric("foo", 2.0, 1)
run.logMetric("foo", 4.0, 2)
run.logMetric("foo", 6.0, 3)
 
new PrintWriter("/tmp/output.txt") { write("Hello, world!") ; close }
run.logArtifact(Paths.get("/tmp/output.txt"))
run.endRun()


// COMMAND ----------


