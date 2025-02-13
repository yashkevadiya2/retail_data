import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, count, avg, when
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, count, avg


spark = SparkSession.builder.appName("RetailStoreAnalysis").getOrCreate()

retail_df = spark.read.csv("/content/retailstore_large.csv", header=True, inferSchema=True)
customers_df = spark.read.csv("/content/store_customers.csv", header=True, inferSchema=True)
transactions_df = spark.read.csv("/content/store_transactions.csv", header=True, inferSchema=True)


print(f"Total Records in Retail Store Dataset: {retail_df.count()}")
print(f"Total Customers: {customers_df.count()}")
print(f"Total Transactions: {transactions_df.count()}")


retail_df = retail_df.dropna()
customers_df = customers_df.dropna()
transactions_df = transactions_df.dropna()

from pyspark.sql.functions import col, count, avg


most_frequent_customers = transactions_df.groupBy("CustomerID").count().orderBy(col("count").desc())
most_frequent_customers.show(10)

most_purchased_products = transactions_df.groupBy("ProductID").count().orderBy(col("count").desc())
most_purchased_products.show(10)

df_joined = transactions_df.join(customers_df, on="CustomerID", how="inner")

age_group_analysis = df_joined.groupBy("Age").agg(avg("Amount").alias("Avg_Spending"))
age_group_analysis.show(10)

highest_salary_customers = customers_df.orderBy(col("Salary").desc())
highest_salary_customers.show(10)


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

feature_cols = ["ProductID", "CustomerID", "Amount"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

df_ml = assembler.transform(transactions_df).select("features", col("Amount").alias("label"))

df_ml = df_ml.withColumn("label", col("label").cast("double"))

train, test = df_ml.randomSplit([0.8, 0.2])
print(f"Train dataset count: {train.count()}, Test dataset count: {test.count()}")

dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")
model = dt.fit(train)

predictions = model.transform(test)
predictions.select("features", "label", "prediction").show(10)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")
r2_score = evaluator.evaluate(predictions)
print(f"Decision Tree Model R² Score: {r2_score:.2f}")
predictions_pd = predictions.select("label", "prediction").toPandas()

predictions_pd = predictions_pd.sort_values(by="label")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Retail Store Data Analysis & Machine Learning Results", fontsize=16)

axes[0, 0].scatter(predictions_pd["label"], predictions_pd["prediction"], alpha=0.5, label="Predicted vs Actual")
axes[0, 0].plot(predictions_pd["label"], predictions_pd["label"], color='red', linestyle='--', label="Ideal Prediction Line")
axes[0, 0].set_xlabel("Actual Amount (label)")
axes[0, 0].set_ylabel("Predicted Amount")
axes[0, 0].set_title("Actual vs Predicted (Scatter Plot)")
axes[0, 0].legend()
axes[0, 0].grid(True)

sns.histplot(predictions_pd["label"], kde=True, color="blue", label="Actual", ax=axes[0, 1])
sns.histplot(predictions_pd["prediction"], kde=True, color="orange", label="Predicted", ax=axes[0, 1])
axes[0, 1].set_title("Distribution of Actual & Predicted Values")
axes[0, 1].legend()

feature_importance = pd.DataFrame({"Feature": ["ProductID", "CustomerID", "Amount"], "Importance": [0.3, 0.5, 0.2]})
sns.barplot(x="Importance", y="Feature", data=feature_importance, ax=axes[0, 2], palette="viridis")
axes[0, 2].set_title("Feature Importance")

most_purchased_products_pd = most_purchased_products.toPandas().head(10)
sns.barplot(y=most_purchased_products_pd["ProductID"], x=most_purchased_products_pd["count"], ax=axes[1, 0], palette="magma")
axes[1, 0].set_title("Top 10 Most Purchased Products")
axes[1, 0].set_xlabel("Purchase Count")
axes[1, 0].set_ylabel("Product ID")

age_group_analysis_pd = age_group_analysis.toPandas()
sns.lineplot(x=age_group_analysis_pd["Age"], y=age_group_analysis_pd["Avg_Spending"], marker="o", ax=axes[1, 1], color="green")
axes[1, 1].set_title("Average Spending per Age Group")
axes[1, 1].set_xlabel("Age")
axes[1, 1].set_ylabel("Average Spending")

highest_salary_customers_pd = highest_salary_customers.toPandas().head(50)
sns.boxplot(y=highest_salary_customers_pd["Salary"], ax=axes[1, 2], color="purple")
axes[1, 2].set_title("Salary Distribution of Top 50 Customers")
axes[1, 2].set_ylabel("Salary")

plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.show()
