// Databricks notebook source

val X = Vector(3.0, 4.0, 5.0)
val Y = Vector(6.0, 9.0, 15.0)

// Step 1: Calculate means
val mean_X = X.sum / X.length
val mean_Y = Y.sum / Y.length

// Step 2: Center the variables
val centered_X = X.map(_ - mean_X)
val centered_Y = Y.map(_ - mean_Y)

// Step 3: Calculate SSD for centered X
val ssdCentered_X = centered_X.map(x => x * x).sum

// Step 4: Calculate covariance of centered X and Y
val covariance_1 = (centered_X, centered_Y).zipped.map(_ * _).sum / (centered_X.length - 1)

// Step 5: Calculate regression weight "w"
val w = covariance_1 / ssdCentered_X

// Step 6: Calculate bias value "b"
val b = mean_Y - w * mean_X

// Step 7: Calculate statistics
val SST = centered_Y.map(y => y * y).sum
val SSR = w * w * ssdCentered_X
val SSE = SST - SSR
val MSE = SSE / (X.length - 2)  
val RMSE = math.sqrt(MSE)
val rSquared = SSR / SST
val r = math.sqrt(rSquared)

println(s"Regression weight (w): $w")
println(s"Bias value (b): $b")
println(s"Total Sum of Squares (SST): $SST")
println(s"Sum of Squares Regression (SSR): $SSR")
println(s"Sum of Squares Error (SSE): $SSE")
println(s"Mean Squared Error (MSE): $MSE")
println(s"Root Mean Squared Error (RMSE): $RMSE")
println(s"Coefficient of Determination (r^2): $rSquared")
println(s"Pearson Correlation Coefficient (r): $r")

  


// COMMAND ----------

// MAGIC %python
// MAGIC import matplotlib.pyplot as plt
// MAGIC import numpy as np
// MAGIC
// MAGIC # Given test vectors
// MAGIC X = np.array([3.0, 4.0, 5.0])
// MAGIC Y = np.array([6.0, 9.0, 15.0])
// MAGIC
// MAGIC # Calculate the regression weight "w" using centered variables
// MAGIC meanX = np.mean(X)
// MAGIC meanY = np.mean(Y)
// MAGIC centeredX = X - meanX
// MAGIC centeredY = Y - meanY
// MAGIC covariance = np.sum(centeredX * centeredY) / (len(X) - 1)
// MAGIC varianceX = np.sum(centeredX ** 2) / (len(X) - 1)
// MAGIC w = covariance / varianceX
// MAGIC
// MAGIC # Calculate the intercept "b"
// MAGIC b = meanY - w * meanX
// MAGIC
// MAGIC # Create a scatter plot of the data points
// MAGIC plt.scatter(X, Y, color='blue', label='Data Points')
// MAGIC
// MAGIC # Plot the regression line
// MAGIC regression_line = w * X + b
// MAGIC plt.plot(X, regression_line, color='red', label='Regression Line')
// MAGIC
// MAGIC # Mark the origin (0, 0)
// MAGIC plt.plot([0], [0], marker='o', markersize=5, color='green', label='Origin (0, 0)')
// MAGIC
// MAGIC # Mark the point (meanX, meanY)
// MAGIC plt.plot([meanX], [meanY], marker='o', markersize=5, color='purple', label='Mean Point')
// MAGIC
// MAGIC # Annotate the components
// MAGIC plt.annotate('x', (meanX, meanY), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='purple')
// MAGIC plt.annotate('y', (meanX, meanY), textcoords="offset points", xytext=(10,0), va='center', fontsize=10, color='purple')
// MAGIC plt.annotate('w', (meanX + 0.5, meanY + w * 0.5), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=10, color='red')
// MAGIC plt.annotate('b', (meanX, meanY + b), textcoords="offset points", xytext=(10,0), va='center', fontsize=10, color='red')
// MAGIC
// MAGIC # Labels and legend
// MAGIC plt.xlabel('X')
// MAGIC plt.ylabel('Y')
// MAGIC plt.title('Statistical Triangle')
// MAGIC plt.legend()
// MAGIC
// MAGIC # Display the plot
// MAGIC plt.grid()
// MAGIC plt.show()

// COMMAND ----------

// MAGIC %python
// MAGIC import matplotlib.pyplot as plt
// MAGIC import numpy as np
// MAGIC
// MAGIC # Given values
// MAGIC w = 2.25
// MAGIC mean_X = 4.0
// MAGIC mean_Y = 10.0
// MAGIC centered_X = np.array([-1.0, 0.0, 1.0])
// MAGIC centered_Y = np.array([-4.0, -1.0, 5.0])
// MAGIC b = 1.0
// MAGIC
// MAGIC # Create a scatter plot of the data points
// MAGIC plt.scatter(centered_X, centered_Y, color='blue', label='Data Points')
// MAGIC
// MAGIC # Annotate the components
// MAGIC plt.annotate('w', (0.5, 0), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='red')
// MAGIC plt.annotate('x', (0.5, 0), textcoords="offset points", xytext=(-15, -15), ha='center', fontsize=10, color='blue')
// MAGIC plt.annotate('y', (0, 1), textcoords="offset points", xytext=(-15, 0), va='center', fontsize=10, color='blue')
// MAGIC plt.annotate('e', (0.5, 0), textcoords="offset points", xytext=(-15, -25), ha='center', fontsize=10, color='green')
// MAGIC
// MAGIC # Identify the legs (x-axis and y-axis)
// MAGIC plt.annotate('x-axis', (1.2, 0), textcoords="offset points", xytext=(0,0), ha='center', fontsize=10, color='blue')
// MAGIC plt.annotate('y-axis', (0, 6), textcoords="offset points", xytext=(0,0), va='center', fontsize=10, color='blue')
// MAGIC
// MAGIC # Labels and legend
// MAGIC plt.xlabel('X')
// MAGIC plt.ylabel('Y')
// MAGIC plt.title('Statistical Triangle with Labeled Components')
// MAGIC plt.legend()
// MAGIC
// MAGIC # Display the plot
// MAGIC plt.grid()
// MAGIC plt.axhline(0, color='black', linewidth=0.5)
// MAGIC plt.axvline(0, color='black', linewidth=0.5)
// MAGIC plt.show()

// COMMAND ----------

val X = Vector(3.0, 4.0, 5.0)
val Y = Vector(6.0, 9.0, 15.0)

// Calculate means
val mean_X = X.sum / X.length
val mean_Y = Y.sum / Y.length

// Center the variables
val centered_X = X.map(_ - mean_X)
val centered_Y = Y.map(_ - mean_Y)

// Calculate SSD for centered X
val ssdCentered_X = centered_X.map(x => x * x).sum

// Calculate covariance of centered X and Y
val covariance_1 = (centered_X, centered_Y).zipped.map(_ * _).sum / (centered_X.length - 1)

// Calculate regression weight "w"
val w = covariance_1 / ssdCentered_X

// Calculate bias value "b"
val b = mean_Y - w * mean_X

// Calculate statistics
val SST = centered_Y.map(y => y * y).sum
val SSR = w * w * ssdCentered_X
val SSE = SST - SSR
val MSE = SSE / (X.length - 2)
val RMSE = math.sqrt(MSE)
val rSquared = SSR / SST
val r = math.sqrt(rSquared)

println(s"Regression weight (w): $w")
println(s"Bias value (b): $b")
println(s"Total Sum of Squares (SST): $SST")
println(s"Sum of Squares Regression (SSR): $SSR")
println(s"Sum of Squares Error (SSE): $SSE")
println(s"Mean Squared Error (MSE): $MSE")
println(s"Root Mean Squared Error (RMSE): $RMSE")
println(s"Coefficient of Determination (r^2): $rSquared")
println(s"Pearson Correlation Coefficient (r): $r")

// COMMAND ----------

// MAGIC %python
// MAGIC import matplotlib.pyplot as plt
// MAGIC import numpy as np
// MAGIC
// MAGIC # Given test vectors
// MAGIC X = np.array([3.0, 4.0, 5.0])
// MAGIC Y = np.array([6.0, 9.0, 15.0])
// MAGIC
// MAGIC # Calculate the regression weight "w" using centered variables
// MAGIC meanX = np.mean(X)
// MAGIC meanY = np.mean(Y)
// MAGIC centeredX = X - meanX
// MAGIC centeredY = Y - meanY
// MAGIC covariance = np.sum(centeredX * centeredY) / (len(X) - 1)
// MAGIC varianceX = np.sum(centeredX ** 2) / (len(X) - 1)
// MAGIC w = covariance / varianceX
// MAGIC
// MAGIC # Calculate the intercept "b"
// MAGIC b = meanY - w * meanX
// MAGIC
// MAGIC # Create a scatter plot of the data points
// MAGIC plt.scatter(X, Y, color='blue', label='Data Points')
// MAGIC
// MAGIC # Plot the regression line
// MAGIC regression_line = w * X + b
// MAGIC plt.plot(X, regression_line, color='red', label='Regression Line')
// MAGIC
// MAGIC # Mark the origin (0, 0)
// MAGIC plt.plot([0], [0], marker='o', markersize=5, color='green', label='Origin (0, 0)')
// MAGIC
// MAGIC # Mark the point (meanX, meanY)
// MAGIC plt.plot([meanX], [meanY], marker='o', markersize=5, color='purple', label='Mean Point')
// MAGIC
// MAGIC # Annotate the components
// MAGIC plt.annotate('x', (meanX, meanY), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10, color='purple')
// MAGIC plt.annotate('y', (meanX, meanY), textcoords="offset points", xytext=(10,0), va='center', fontsize=10, color='purple')
// MAGIC plt.annotate('w', (meanX + 0.5, meanY + w * 0.5), textcoords="offset points", xytext=(-15,10), ha='center', fontsize=10, color='red')
// MAGIC plt.annotate('b', (meanX, meanY + b), textcoords="offset points", xytext=(10,0), va='center', fontsize=10, color='red')
// MAGIC
// MAGIC # Labels and legend
// MAGIC plt.xlabel('X')
// MAGIC plt.ylabel('Y')
// MAGIC plt.title('Statistical Triangle')
// MAGIC plt.legend()
// MAGIC
// MAGIC # Display the plot
// MAGIC plt.grid()
// MAGIC plt.show()
// MAGIC
// MAGIC

// COMMAND ----------




// COMMAND ----------


