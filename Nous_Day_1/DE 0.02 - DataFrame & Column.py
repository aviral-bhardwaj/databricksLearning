# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-ab2602b5-4183-4f33-8063-cfc03fcb1425
# MAGIC %md
# MAGIC # DataFrame & Column
# MAGIC ##### Objectives
# MAGIC 1. Construct columns
# MAGIC 1. Subset columns
# MAGIC 1. Add or replace columns
# MAGIC 1. Subset rows
# MAGIC 1. Sort rows
# MAGIC
# MAGIC ##### Methods
# MAGIC - <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html" target="_blank">DataFrame</a>: **`select`**, **`selectExpr`**, **`drop`**, **`withColumn`**, **`withColumnRenamed`**, **`filter`**, **`distinct`**, **`limit`**, **`sort`**
# MAGIC - <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/column.html" target="_blank">Column</a>: **`alias`**, **`isin`**, **`cast`**, **`isNotNull`**, **`desc`**, operators

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit

spark = SparkSession.builder.getOrCreate()

data = [
    (1, "Alice",   "IT",      80000, "IN", 5,  True),
    (2, "Bob",     "IT",      75000, "US", 4,  True),
    (3, "Charlie", "HR",      60000, "IN", 3,  False),
    (4, "David",   "Finance", 90000, "UK", 7,  True),
    (5, "Eve",     "HR",      65000, "IN", 4,  True),
    (6, "Frank",   "IT",      70000, "DE", 2,  False),
    (7, "Grace",   "Finance", 95000, "US", 8,  True),
    (8, "Heidi",   "IT",      72000, "IN", 3,  True),
    (9, "Ivan",    "Sales",   68000, "IN", 2,  False),
    (10,"Judy",    "Sales",   73000, "US", 4,  True),
    (11,"Ken",     "IT",      81000, "IN", 5,  True),
    (12,"Leo",     "IT",      82000, "IN", 6,  None),
    (13,"Mallory", "Finance", 99000, "US", 9,  True),
    (14,"Niaj",    "Sales",   66000, "IN", 1,  False),
    (15,"Olivia",  "HR",      64000, "CA", 3,  True),
    (16,"Peggy",   "IT",      77000, "IN", 4,  True),
    (17,"Quentin", "Finance", 93000, "IN", 6,  True),
    (18,"Rupert",  "Sales",   71000, "UK", 3,  False),
    (19,"Sybil",   "IT",      85000, "US", 7,  True),
    (20,"Trent",   "HR",      60500, "IN", 2,  None),
]

schema = "emp_id INT, name STRING, dept STRING, salary INT, country STRING, experience INT, is_active BOOLEAN"

emp_df = spark.createDataFrame(data, schema=schema)



# COMMAND ----------

# DBTITLE 0,--i18n-ef990348-e991-4edb-bf45-84de46a34759
# MAGIC %md
# MAGIC
# MAGIC Let's use the BedBricks events dataset.

# COMMAND ----------

emp_df.display()

# COMMAND ----------

# DBTITLE 0,--i18n-4ea9a278-1eb6-45ad-9f96-34e0fd0da553
# MAGIC %md
# MAGIC
# MAGIC ## Column Expressions
# MAGIC
# MAGIC A <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/column.html" target="_blank">Column</a> is a logical construction that will be computed based on the data in a DataFrame using an expression
# MAGIC
# MAGIC Construct a new Column based on existing columns in a DataFrame

# COMMAND ----------

from pyspark.sql.functions import col

print(emp_df.name)
print(emp_df["name"])
print(col("name"))

# COMMAND ----------

# DBTITLE 0,--i18n-d87b8303-8f78-416e-99b0-b037caf2107a
# MAGIC %md
# MAGIC Scala supports an additional syntax for creating a new Column based on existing columns in a DataFrame

# COMMAND ----------



# COMMAND ----------

# DBTITLE 0,--i18n-64238a77-0877-4bd4-af46-a9a8bd4763c6
# MAGIC %md
# MAGIC
# MAGIC ### Column Operators and Methods
# MAGIC | Method | Description |
# MAGIC | --- | --- |
# MAGIC | \*, + , <, >= | Math and comparison operators |
# MAGIC | ==, != | Equality and inequality tests (Scala operators are **`===`** and **`=!=`**) |
# MAGIC | alias | Gives the column an alias |
# MAGIC | cast, astype | Casts the column to a different data type |
# MAGIC | isNull, isNotNull, isNan | Is null, is not null, is NaN |
# MAGIC | asc, desc | Returns a sort expression based on ascending/descending order of the column |

# COMMAND ----------

# DBTITLE 0,--i18n-6d68007e-3dbf-4f18-bde4-6990299ef086
# MAGIC %md
# MAGIC
# MAGIC Create complex expressions with existing columns, operators, and methods.

# COMMAND ----------

col("salary") + col("experience")
col("emp_id").desc()
(col("salary") * 100).cast("int")

# COMMAND ----------

# DBTITLE 0,--i18n-7c1c0688-8f9f-4247-b8b8-bb869414b276
# MAGIC %md
# MAGIC Here's an example of using these column expressions in the context of a DataFrame

# COMMAND ----------

rev_df = (
    emp_df
    .filter(col("salary").isNotNull())
    .withColumn("purchase_revenue", (col("salary") * 100).cast("int"))
    .withColumn("avg_purchase_revenue", col("salary") / col("experience"))
    .sort(col("avg_purchase_revenue").desc())
)

display(rev_df)

# COMMAND ----------

# DBTITLE 0,--i18n-7ba60230-ecd3-49dd-a4c8-d964addc6692
# MAGIC %md
# MAGIC
# MAGIC ## DataFrame Transformation Methods
# MAGIC | Method | Description |
# MAGIC | --- | --- |
# MAGIC | **`select`** | Returns a new DataFrame by computing given expression for each element |
# MAGIC | **`drop`** | Returns a new DataFrame with a column dropped |
# MAGIC | **`withColumnRenamed`** | Returns a new DataFrame with a column renamed |
# MAGIC | **`withColumn`** | Returns a new DataFrame by adding a column or replacing the existing column that has the same name |
# MAGIC | **`filter`**, **`where`** | Filters rows using the given condition |
# MAGIC | **`sort`**, **`orderBy`** | Returns a new DataFrame sorted by the given expressions |
# MAGIC | **`dropDuplicates`**, **`distinct`** | Returns a new DataFrame with duplicate rows removed |
# MAGIC | **`limit`** | Returns a new DataFrame by taking the first n rows |
# MAGIC | **`groupBy`** | Groups the DataFrame using the specified columns, so we can run aggregation on them |

# COMMAND ----------

# DBTITLE 0,--i18n-3e95eb92-30e4-44aa-8ee0-46de94c2855e
# MAGIC %md
# MAGIC
# MAGIC ### Subset columns
# MAGIC Use DataFrame transformations to subset columns

# COMMAND ----------

# DBTITLE 0,--i18n-987cfd99-8e06-447f-b1c7-5f104cd5ed2f
# MAGIC %md
# MAGIC
# MAGIC #### **`select()`**
# MAGIC Selects a list of columns or column based expressions

# COMMAND ----------

two_df = emp_df.select("emp_id", "name")
display(two_df)

# COMMAND ----------

from pyspark.sql.functions import col

locations_df = emp_df.select(
    "emp_id", 
    col("country").alias("city"), 
    col("dept").alias("state")
)
display(locations_df)

# COMMAND ----------

# DBTITLE 0,--i18n-8d556f84-bfcd-436a-a3dd-893143ce620e
# MAGIC %md
# MAGIC
# MAGIC #### **`selectExpr()`**
# MAGIC Selects a list of SQL expressions

# COMMAND ----------

apple_df = emp_df.selectExpr("emp_id", "country in ('US', 'IN') as apple_user")
display(apple_df)

# COMMAND ----------

# DBTITLE 0,--i18n-452f7fb3-3866-4835-827f-6d359f364046
# MAGIC %md
# MAGIC
# MAGIC #### **`drop()`**
# MAGIC Returns a new DataFrame after dropping the given column, specified as a string or Column object
# MAGIC
# MAGIC Use strings to specify multiple columns

# COMMAND ----------

anonymous_df = emp_df.drop("emp_id", "country", "dept")
display(anonymous_df)

# COMMAND ----------

no_sales_df = emp_df.drop("salary")
display(no_sales_df)

# COMMAND ----------

# DBTITLE 0,--i18n-b11609a3-11d5-453b-b713-15131b277066
# MAGIC %md
# MAGIC
# MAGIC ### Add or replace columns
# MAGIC Use DataFrame transformations to add or replace columns

# COMMAND ----------

# DBTITLE 0,--i18n-f29a47d9-9567-40e5-910b-73c640cc61ca
# MAGIC %md
# MAGIC
# MAGIC #### **`withColumn()`**
# MAGIC Returns a new DataFrame by adding a column or replacing an existing column that has the same name.

# COMMAND ----------

mobile_emp_df = emp_df.withColumn("mobile", col("country").isin("US", "IN"))
display(mobile_emp_df)

# COMMAND ----------

purchase_quantity_emp_df = emp_df.withColumn("purchase_quantity", col("salary").cast("int"))
purchase_quantity_emp_df.printSchema()

# COMMAND ----------

# DBTITLE 0,--i18n-969c0d9f-202f-405a-8c66-ef29076b48fc
# MAGIC %md
# MAGIC
# MAGIC #### **`withColumnRenamed()`**
# MAGIC Returns a new DataFrame with a column renamed.

# COMMAND ----------

location_emp_df = emp_df.withColumnRenamed("country", "location")
display(location_emp_df)

# COMMAND ----------

# DBTITLE 0,--i18n-23b0a9ef-58d5-4973-a610-93068a998d5e
# MAGIC %md
# MAGIC
# MAGIC ### Subset Rows
# MAGIC Use DataFrame transformations to subset rows

# COMMAND ----------

# DBTITLE 0,--i18n-4ada6444-7345-41f7-aaa2-1de2d729483f
# MAGIC %md
# MAGIC
# MAGIC #### **`filter()`**
# MAGIC Filters rows using the given SQL expression or column based condition.
# MAGIC
# MAGIC ##### Alias: **`where`**

# COMMAND ----------

purchases_emp_df = emp_df.filter("salary > 0")
display(purchases_emp_df)

# COMMAND ----------

revenue_emp_df = emp_df.filter(col("salary").isNotNull())
display(revenue_emp_df)

# COMMAND ----------

android_emp_df = emp_df.filter((col("country") != "US") & (col("dept") == "IT"))
display(android_emp_df)

# COMMAND ----------

# DBTITLE 0,--i18n-4d6a79eb-3989-43e1-8c28-5b976a513f5f
# MAGIC %md
# MAGIC
# MAGIC #### **`dropDuplicates()`**
# MAGIC Returns a new DataFrame with duplicate rows removed, optionally considering only a subset of columns.
# MAGIC
# MAGIC ##### Alias: **`distinct`**

# COMMAND ----------

display(emp_df.distinct())

# COMMAND ----------

distinct_emp_df = emp_df.dropDuplicates(["is_active"])
display(distinct_emp_df)

# COMMAND ----------

# DBTITLE 0,--i18n-433c57f4-ce40-48c9-8d04-d3a13c398082
# MAGIC %md
# MAGIC
# MAGIC #### **`limit()`**
# MAGIC Returns a new DataFrame by taking the first n rows.

# COMMAND ----------

limit_df = emp_df.limit(4)
display(limit_df)

# COMMAND ----------

# DBTITLE 0,--i18n-d4117305-e742-497e-964d-27a7b0c395cd
# MAGIC %md
# MAGIC
# MAGIC ### Sort rows
# MAGIC Use DataFrame transformations to sort rows

# COMMAND ----------

# DBTITLE 0,--i18n-16b3c7fe-b5f2-4564-9e8e-4f677777c50c
# MAGIC %md
# MAGIC
# MAGIC #### **`sort()`**
# MAGIC Returns a new DataFrame sorted by the given columns or expressions.
# MAGIC
# MAGIC ##### Alias: **`orderBy`**

# COMMAND ----------

increase_salary_df = emp_df.sort("salary")
display(increase_salary_df)

# COMMAND ----------

decrease_salary_df = emp_df.sort(col("salary").desc())
display(decrease_salary_df)

# COMMAND ----------

increase_sessions_emp_df = emp_df.orderBy(["salary", "experience"])
display(increase_sessions_emp_df)

# COMMAND ----------

decrease_sessions_emp_df = emp_df.sort(col("salary").desc(), col("experience"))
display(decrease_sessions_emp_df)

# COMMAND ----------

# DBTITLE 0,--i18n-555c663e-3f62-4478-9d76-c9ee090beca1
# MAGIC %md
# MAGIC
# MAGIC Run the following cell to delete the tables and files associated with this lesson.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
