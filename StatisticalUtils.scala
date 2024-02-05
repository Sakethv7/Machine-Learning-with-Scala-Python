// Databricks notebook source

object StatUtils {
  type D = Double
  type I = Integer
  type V = Vector[D]

  def mean(v: V): D = v.sum / v.size

  def centered(v: V): V = {
    val meanValue = mean(v)
    v.map(_ - meanValue)
  }

  def dot(v: V, w: V): D = (v zip w).map { case (x, y) => x * y }.sum

  def norm(v: V): D = math.sqrt(v.map(x => x * x).sum)

  def variance(v: V): D = {
    val vMean = mean(v)
    v.map(x => math.pow(x - vMean, 2)).sum / v.size
  }

  def std(v: V): D = math.sqrt(variance(v))

  def correlation(v: V, w: V): D = {
    val vCentered = centered(v)
    val wCentered = centered(w)
    dot(vCentered, wCentered) / (norm(vCentered) * norm(wCentered))
  }
}

// COMMAND ----------

val v1 = Vector(3.0, 4.0, 5.0)
val f1 = Vector(6.0, 9.0, 15.0)

println(s"Mean of v1: ${StatUtils.mean(v1)}")
println(s"Centered v1: ${StatUtils.centered(v1)}")
println(s"Dot product of v1 and f1: ${StatUtils.dot(v1, f1)}")
println(s"Norm of v1: ${StatUtils.norm(v1)}")
println(s"Variance of v1: ${StatUtils.variance(v1)}")
println(s"Standard deviation of v1: ${StatUtils.std(v1)}")
println(s"Correlation between v1 and f1: ${StatUtils.correlation(v1, f1)}")


// COMMAND ----------


