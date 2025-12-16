# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# DBTITLE 0,--i18n-ad7af192-ab00-41a3-b683-5de4856cacb0
# MAGIC %md
# MAGIC # Spark SQL
# MAGIC
# MAGIC Demonstrate fundamental concepts in Spark SQL using the DataFrame API.
# MAGIC
# MAGIC ##### Objectives
# MAGIC 1. Run a SQL query
# MAGIC 1. Create a DataFrame from a table
# MAGIC 1. Write the same query using DataFrame transformations
# MAGIC 1. Trigger computation with DataFrame actions
# MAGIC 1. Convert between DataFrames and SQL
# MAGIC
# MAGIC ##### Methods
# MAGIC - <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/spark_session.html" target="_blank">SparkSession</a>: **`sql`**, **`table`**
# MAGIC - <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/dataframe.html" target="_blank">DataFrame</a>:
# MAGIC   - Transformations:  **`select`**, **`where`**, **`orderBy`**
# MAGIC   - Actions: **`show`**, **`count`**, **`take`**
# MAGIC   - Other methods: **`printSchema`**, **`schema`**, **`createOrReplaceTempView`**

# COMMAND ----------

# MAGIC %sh
# MAGIC pwd

# COMMAND ----------



# COMMAND ----------

#%run ./Includes/Classroom-Setup-00.01
#%run /Workspace/Users/naveenktn@nousinfo.com/databricksLearning/Nous_Day_1/Includes/Classroom-Setup-00.01


# COMMAND ----------

# DBTITLE 0,--i18n-3ad6c2cb-bfa4-4af5-b637-ba001a9ef54b
# MAGIC %md
# MAGIC
# MAGIC ## Multiple Interfaces
# MAGIC Spark SQL is a module for structured data processing with multiple interfaces.
# MAGIC
# MAGIC We can interact with Spark SQL in two ways:
# MAGIC 1. Executing SQL queries
# MAGIC 1. Working with the DataFrame API.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Transformations and Actions in Spark
# MAGIC
# MAGIC **Transformation**  
# MAGIC A transformation is an operation on a DataFrame or RDD that produces a new DataFrame or RDD. Transformations are *lazy*, meaning they are not executed until an action is called.  
# MAGIC - **Narrow Transformation**: Data required to compute the records in a single partition resides in a single partition of the parent. Less shuffling, more efficient.  
# MAGIC   *Example*: `df.select("column")`, `df.where("column > 10")`
# MAGIC - **Wide Transformation**: Data from multiple partitions may be required, causing shuffling across the cluster. More costly.  
# MAGIC   *Example*: `df.groupBy("column").count()`, `df.orderBy("column")`
# MAGIC
# MAGIC **Action**  
# MAGIC An action triggers the execution of transformations and returns a result to the driver or writes data to external storage. Actions are generally more costly because they require computation.  
# MAGIC *Example*: `df.show()`, `df.collect()`, `df.count()`
# MAGIC
# MAGIC **Lazy Evaluation**  
# MAGIC Spark uses lazy evaluation for transformations, meaning computation is deferred until an action is called. This allows Spark to optimize the execution plan and reduce costs.
# MAGIC
# MAGIC **Example:**
# MAGIC
# MAGIC python
# MAGIC # Transformation (lazy)
# MAGIC filtered_df = df.where("price > 100")
# MAGIC
# MAGIC # Action (triggers computation)
# MAGIC filtered_df.show()

# COMMAND ----------

# DBTITLE 0,--i18n-236a9dcf-8e89-4b08-988a-67c3ca31bb71
# MAGIC %md
# MAGIC **Method 1: Executing SQL queries**
# MAGIC
# MAGIC This is a basic SQL query.

# COMMAND ----------


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, DateType
from datetime import date

# Initialize Spark Session
spark = SparkSession.builder.appName("DummyOrdersTable").getOrCreate()

# Define schema
schema = StructType([
    StructField("order_id", IntegerType(), False),
    StructField("customer_name", StringType(), True),
    StructField("product", StringType(), True),
    StructField("quantity", IntegerType(), True),
    StructField("price", DoubleType(), True),
    StructField("order_date", DateType(), True)
])

# Create sample data
data = [
    (1, "Alice", "Laptop", 1, 75000.00, date(2025, 12, 10)),
    (2, "Bob", "Smartphone", 2, 30000.00, date(2025, 12, 11)),
    (3, "Charlie", "Headphones", 3, 1500.00, date(2025, 12, 12)),
       (4, "David", "Monitor", 1, 12000.00, date(2025, 12, 13)),
    (5, "Eva", "Keyboard", 2, 2000.00, date(2025, 12, 14))
]

# Create DataFrame
orders_df = spark.createDataFrame(data, schema)

# Show DataFrame
 

# COMMAND ----------

#orders_df.display()
display(orders_df)



# COMMAND ----------

orders_df.createOrReplaceTempView("orders")


# COMMAND ----------

# MAGIC %md
# MAGIC #Using SQL Command
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from orders

# COMMAND ----------

# MAGIC %md 
# MAGIC #Using SparkSQl
# MAGIC

# COMMAND ----------

spark.sql('select * from orders').show()

# COMMAND ----------

# MAGIC %md
# MAGIC #Using SelectExpr
# MAGIC

# COMMAND ----------

display(orders_df.selectExpr("*"))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from orders where price <=1500

# COMMAND ----------

display(spark.table("orders").select ("customer_name","product","price"))


# COMMAND ----------

orders_df.printSchema()

# COMMAND ----------

# MAGIC
# MAGIC %sql
# MAGIC SELECT name, price
# MAGIC FROM products
# MAGIC WHERE price < 200
# MAGIC ORDER BY price

# COMMAND ----------

# MAGIC %md
# MAGIC #Using SQL Command

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select * from orders

# COMMAND ----------

# MAGIC %md
# MAGIC #Using SparkSQL

# COMMAND ----------

spark.sql('select * from orders').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##Using SELECTEXPR

# COMMAND ----------

display(orders_df.selectExpr("*"))

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select * from orders where price <= 1500

# COMMAND ----------

display(spark.table("orders").select("customer_name", "product","price")
             .where("price <= 2000")
             .orderBy("price")
            )

# COMMAND ----------

orders_df.printSchema()

# COMMAND ----------



# COMMAND ----------

# DBTITLE 0,--i18n-bb02bfff-cf98-4639-af21-76bec5c8d95b
# MAGIC %md
# MAGIC
# MAGIC ## Query Execution
# MAGIC We can express the same query using any interface. The Spark SQL engine generates the same query plan used to optimize and execute on our Spark cluster.
# MAGIC
# MAGIC ![query execution engine](https://files.training.databricks.com/images/aspwd/spark_sql_query_execution_engine.png)
# MAGIC
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_32.png" alt="Note"> Resilient Distributed Datasets (RDDs) are the low-level representation of datasets processed by a Spark cluster. In early versions of Spark, you had to write <a href="https://spark.apache.org/docs/latest/rdd-programming-guide.html" target="_blank">code manipulating RDDs directly</a>. In modern versions of Spark you should instead use the higher-level DataFrame APIs, which Spark automatically compiles into low-level RDD operations.

# COMMAND ----------

# DBTITLE 0,--i18n-fbaea5c1-fefc-4b3b-a645-824ffa77bbd5
# MAGIC %md
# MAGIC
# MAGIC ## Spark API Documentation
# MAGIC
# MAGIC To learn how we work with DataFrames in Spark SQL, let's first look at the Spark API documentation.
# MAGIC The main Spark <a href="https://spark.apache.org/docs/latest/" target="_blank">documentation</a> page includes links to API docs and helpful guides for each version of Spark.
# MAGIC
# MAGIC The <a href="https://spark.apache.org/docs/latest/api/scala/org/apache/spark/index.html" target="_blank">Scala API</a> and <a href="https://spark.apache.org/docs/latest/api/python/index.html" target="_blank">Python API</a> are most commonly used, and it's often helpful to reference the documentation for both languages.
# MAGIC Scala docs tend to be more comprehensive, and Python docs tend to have more code examples.
# MAGIC
# MAGIC #### Navigating Docs for the Spark SQL Module
# MAGIC Find the Spark SQL module by navigating to **`org.apache.spark.sql`** in the Scala API or **`pyspark.sql`** in the Python API.
# MAGIC The first class we'll explore in this module is the **`SparkSession`** class. You can find this by entering "SparkSession" in the search bar.

# COMMAND ----------

# DBTITLE 0,--i18n-24790eda-96df-49bb-af34-b1ed839fa80a
# MAGIC %md
# MAGIC ## SparkSession
# MAGIC The **`SparkSession`** class is the single entry point to all functionality in Spark using the DataFrame API.
# MAGIC
# MAGIC In Databricks notebooks, the SparkSession is created for you, stored in a variable called **`spark`**.

# COMMAND ----------

# DBTITLE 0,--i18n-4f5934fb-12b9-4bf2-b821-5ab17d627309
# MAGIC %md
# MAGIC
# MAGIC The example from the beginning of this lesson used the SparkSession method **`table`** to create a DataFrame from the **`products`** table. Let's save this in the variable **`products_df`**.

# COMMAND ----------

# DBTITLE 0,--i18n-f9968eff-ed08-4ed6-9fe7-0252b94bf50a
# MAGIC %md
# MAGIC Below are several additional methods we can use to create DataFrames. All of these can be found in the <a href="https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.SparkSession.html" target="_blank">documentation</a> for **`SparkSession`**.
# MAGIC
# MAGIC #### **`SparkSession`** Methods
# MAGIC | Method | Description |
# MAGIC | --- | --- |
# MAGIC | sql | Returns a DataFrame representing the result of the given query |
# MAGIC | table | Returns the specified table as a DataFrame |
# MAGIC | read | Returns a DataFrameReader that can be used to read data in as a DataFrame |
# MAGIC | range | Create a DataFrame with a column containing elements in a range from start to end (exclusive) with step value and number of partitions |
# MAGIC | createDataFrame | Creates a DataFrame from a list of tuples, primarily used for testing |

# COMMAND ----------

# DBTITLE 0,--i18n-2277e250-91f9-489a-940b-97d17e75c7f5
# MAGIC %md
# MAGIC
# MAGIC Let's use a SparkSession method to run SQL.

# COMMAND ----------

# DBTITLE 0,--i18n-f2851702-3573-4cb4-9433-ec31d4ceb0f2
# MAGIC %md
# MAGIC
# MAGIC ## DataFrames
# MAGIC Recall that expressing our query using methods in the DataFrame API returns results in a DataFrame. Let's store this in the variable **`budget_df`**.
# MAGIC
# MAGIC A **DataFrame** is a distributed collection of data grouped into named columns.

# COMMAND ----------

# DBTITLE 0,--i18n-d538680a-1d7a-433c-9715-7fd975d4427b
# MAGIC %md
# MAGIC
# MAGIC We can use **`display()`** to output the results of a dataframe.

# COMMAND ----------

# DBTITLE 0,--i18n-ea532d26-a607-4860-959a-00a2eca34305
# MAGIC %md
# MAGIC
# MAGIC The **schema** defines the column names and types of a dataframe.
# MAGIC
# MAGIC Access a dataframe's schema using the **`schema`** attribute.

# COMMAND ----------

# DBTITLE 0,--i18n-4212166a-a200-44b5-985c-f7f1b33709a3
# MAGIC %md
# MAGIC
# MAGIC View a nicer output for this schema using the **`printSchema()`** method.

# COMMAND ----------

# DBTITLE 0,--i18n-7ad577db-093a-40fb-802e-99bbc5a4435b
# MAGIC %md
# MAGIC
# MAGIC ## Transformations
# MAGIC When we created **`budget_df`**, we used a series of DataFrame transformation methods e.g. **`select`**, **`where`**, **`orderBy`**.
# MAGIC
# MAGIC <strong><code>products_df  
# MAGIC &nbsp;  .select("name", "price")  
# MAGIC &nbsp;  .where("price < 200")  
# MAGIC &nbsp;  .orderBy("price")  
# MAGIC </code></strong>
# MAGIC     
# MAGIC Transformations operate on and return DataFrames, allowing us to chain transformation methods together to construct new DataFrames.
# MAGIC However, these operations can't execute on their own, as transformation methods are **lazily evaluated**.
# MAGIC
# MAGIC Running the following cell does not trigger any computation.

# COMMAND ----------

# DBTITLE 0,--i18n-56f40b55-842f-44cf-b34a-b0fd17a962d4
# MAGIC %md
# MAGIC
# MAGIC ## Actions
# MAGIC Conversely, DataFrame actions are methods that **trigger computation**.
# MAGIC Actions are needed to trigger the execution of any DataFrame transformations.
# MAGIC
# MAGIC The **`show`** action causes the following cell to execute transformations.

# COMMAND ----------

# DBTITLE 0,--i18n-6f574091-0026-4dd2-9763-c6d4c3b9c4fe
# MAGIC %md
# MAGIC
# MAGIC Below are several examples of <a href="https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql.html#dataframe-apis" target="_blank">DataFrame</a> actions.
# MAGIC
# MAGIC ### DataFrame Action Methods
# MAGIC | Method | Description |
# MAGIC | --- | --- |
# MAGIC | show | Displays the top n rows of DataFrame in a tabular form |
# MAGIC | count | Returns the number of rows in the DataFrame |
# MAGIC | describe,  summary | Computes basic statistics for numeric and string columns |
# MAGIC | first, head | Returns the the first row |
# MAGIC | collect | Returns an array that contains all rows in this DataFrame |
# MAGIC | take | Returns an array of the first n rows in the DataFrame |

# COMMAND ----------

# DBTITLE 0,--i18n-7e725f41-43bc-4e56-9c44-d46becd375a0
# MAGIC %md
# MAGIC **`count`** returns the number of records in a DataFrame.

# COMMAND ----------

# DBTITLE 0,--i18n-12ea69d5-587e-4953-80d9-81955eeb9d7b
# MAGIC %md
# MAGIC **`collect`** returns an array of all rows in a DataFrame.

# COMMAND ----------

# DBTITLE 0,--i18n-983f5da8-b456-42b5-b21c-b6f585c697b4
# MAGIC %md
# MAGIC
# MAGIC ## Convert between DataFrames and SQL

# COMMAND ----------

# DBTITLE 0,--i18n-0b6ceb09-86dc-4cdd-9721-496d01e8737f
# MAGIC %md
# MAGIC **`createOrReplaceTempView`** creates a temporary view based on the DataFrame. The lifetime of the temporary view is tied to the SparkSession that was used to create the DataFrame.

# COMMAND ----------

# DBTITLE 0,--i18n-81ff52ca-2160-4bfd-a78f-b3ba2f8b4933
# MAGIC %md
# MAGIC
# MAGIC Run the following cell to delete the tables and files associated with this lesson.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
