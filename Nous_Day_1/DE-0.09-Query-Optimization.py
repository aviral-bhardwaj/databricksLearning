# Databricks notebook source
# MAGIC %md
# MAGIC # Query Optimization - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates PySpark query optimization techniques:
# MAGIC - **EXPLAIN**: Understand query execution plans
# MAGIC - **CACHING**: Store DataFrames in memory
# MAGIC - **FILTERING**: Apply early filters to reduce data
# MAGIC - **PARTITIONING**: Distribute reads efficiently
# MAGIC
# MAGIC Each technique improves query performance significantly.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, sum, avg, explain
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create Event Log DataFrame
# MAGIC
# MAGIC Creating event log data with:
# MAGIC - `event_id`: Event ID
# MAGIC - `user_id`: User ID
# MAGIC - `event_name`: Type of event (login, click, purchase, etc.)
# MAGIC - `event_timestamp`: When event occurred
# MAGIC - `duration_ms`: Duration in milliseconds
# MAGIC - `status_code`: HTTP status code

# COMMAND ----------

spark = SparkSession.builder.appName("QueryOptimization").getOrCreate()

# Event Log Data (simulating a large dataset)
event_data = [
    (1, 101, "login", "2023-12-01 08:00:00", 150, 200),
    (2, 101, "reviews", "2023-12-01 08:05:00", 3000, 200),
    (3, 101, "click", "2023-12-01 08:10:00", 200, 200),
    (4, 102, "checkout", "2023-12-01 08:15:00", 5000, 200),
    (5, 102, "register", "2023-12-01 08:20:00", 2000, 200),
    (6, 103, "emailcoupon", "2023-12-01 08:25:00", 100, 200),
    (7, 103, "ccinfo", "2023-12-01 08:30:00", 800, 200),
    (8, 104, "delivery", "2023-12-01 08:35:00", 1500, 200),
    (9, 104, "shippinginfo", "2023-12-01 08:40:00", 600, 200),
    (10, 105, "press", "2023-12-01 08:45:00", 50, 200),
    (11, 101, "login", "2023-12-02 09:00:00", 120, 200),
    (12, 102, "reviews", "2023-12-02 09:05:00", 2500, 200),
    (13, 103, "checkout", "2023-12-02 09:10:00", 4500, 200),
    (14, 104, "register", "2023-12-02 09:15:00", 1800, 200),
    (15, 105, "click", "2023-12-02 09:20:00", 180, 200),
    (16, 101, "finalize", "2023-12-02 09:25:00", 300, 200),
    (17, 102, "finalize", "2023-12-02 09:30:00", 350, 200),
    (18, 103, "finalize", "2023-12-02 09:35:00", 320, 200),
    (19, 104, "finalize", "2023-12-02 09:40:00", 380, 200),
    (20, 105, "finalize", "2023-12-02 09:45:00", 290, 200),
]

event_schema = StructType([
    StructField("event_id", IntegerType(), True),
    StructField("user_id", IntegerType(), True),
    StructField("event_name", StringType(), True),
    StructField("event_timestamp", StringType(), True),
    StructField("duration_ms", IntegerType(), True),
    StructField("status_code", IntegerType(), True),
])

df = spark.createDataFrame(event_data, event_schema)

print("===== ORIGINAL EVENT DATA =====")
df.printSchema()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: EXPLAIN - Understanding Query Plans
# MAGIC
# MAGIC **Goal:** Use EXPLAIN to understand how Spark executes queries.
# MAGIC
# MAGIC **Function:** `.explain(extended=True/False)`
# MAGIC - Shows logical and physical execution plans

# COMMAND ----------

print("===== EXPLAIN - NAIVE APPROACH (MULTIPLE FILTERS) =====\n")

# Inefficient: Multiple separate filters
naive_df = df \
    .filter(col("event_name") != "reviews") \
    .filter(col("event_name") != "checkout") \
    .filter(col("event_name") != "register") \
    .filter(col("event_name") != "emailcoupon") \
    .filter(col("event_name") != "ccinfo") \
    .filter(col("event_name") != "delivery") \
    .filter(col("event_name") != "shippinginfo") \
    .filter(col("event_name") != "press")

naive_df.explain(True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: OPTIMIZED Query Plan
# MAGIC
# MAGIC **Goal:** Write efficient queries using combined filters.
# MAGIC
# MAGIC **Best Practice:** Combine filters in single WHERE clause

# COMMAND ----------

print("===== OPTIMIZED APPROACH (COMBINED FILTER) =====\n")

# Better: Combined filter with AND conditions
optimized_df = df.filter(
    (col("event_name").isNotNull()) & 
    (col("event_name") != "reviews") &
    (col("event_name") != "checkout") &
    (col("event_name") != "register") &
    (col("event_name") != "emailcoupon") &
    (col("event_name") != "ccinfo") &
    (col("event_name") != "delivery") &
    (col("event_name") != "shippinginfo") &
    (col("event_name") != "press")
)

optimized_df.explain(True)

print(f"\nNaive result: {naive_df.count()} rows")
print(f"Optimized result: {optimized_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Filtering Early - Best Practice
# MAGIC
# MAGIC **Goal:** Apply filters BEFORE aggregations to reduce data.

# COMMAND ----------

print("===== FILTERING EARLY (KEY OPTIMIZATION) =====\n")

# INEFFICIENT: Aggregate first, then filter
# expensive_df = df.groupBy("user_id").agg(count("*")).filter(col("count(1)") > 2)

# EFFICIENT: Filter first, then aggregate
efficient_df = df.filter(
    col("event_name").isin(["login", "click", "finalize"])
).groupBy("user_id").agg(
    count("*").alias("event_count"),
    avg("duration_ms").alias("avg_duration")
)

print("Early filter reduces data size before aggregation:")
display(efficient_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: CACHING - Store DataFrames in Memory
# MAGIC
# MAGIC **Goal:** Cache DataFrames used multiple times for faster access.
# MAGIC
# MAGIC **Function:** `.cache()` or `.persist()`
# MAGIC - First action materializes to memory
# MAGIC - Subsequent actions read from cache

# COMMAND ----------

print("===== CACHING - MEMORY STORAGE =====\n")

# Create a filtered DataFrame
base_df = df.filter(col("event_name").isin(["login", "click", "finalize"]))

# CACHE for reuse
base_df.cache()

print(f"Cached DataFrame has {base_df.count()} rows\n")

# First use of cache - materializes
result1 = base_df.filter(col("duration_ms") > 200)
print(f"Query 1 (from cache): {result1.count()} rows")

# Second use of cache - reads from memory
result2 = base_df.filter(col("user_id") == 101)
print(f"Query 2 (from cache): {result2.count()} rows")

# Third use of cache - still from memory
result3 = base_df.groupBy("user_id").agg(count("*"))
print(f"Query 3 (from cache): {result3.count()} rows\n")

print("Benefit: Cache speeds up multiple queries on same data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Unpersist - Release Cache
# MAGIC
# MAGIC **Goal:** Remove cached DataFrame when done to free memory.
# MAGIC
# MAGIC **Function:** `.unpersist()`

# COMMAND ----------

print("===== UNPERSIST - RELEASE CACHE =====\n")

# Remove from cache
base_df.unpersist()

print("✓ Cache released, memory freed for other operations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Practical Optimization Pattern
# MAGIC
# MAGIC **Goal:** Demonstrate complete optimization workflow.

# COMMAND ----------

print("===== COMPLETE OPTIMIZATION PATTERN =====\n")

# Step 1: Filter early (reduces data)
filtered_df = df.filter(
    (col("event_name") != "finalize") &
    (col("duration_ms") > 100) &
    (col("status_code") == 200)
)

# Step 2: Cache for multiple downstream uses
filtered_df.cache()

# Step 3: Perform multiple analyses
user_summary = filtered_df.groupBy("user_id").agg(
    count("*").alias("total_events"),
    avg("duration_ms").alias("avg_duration")
)

event_summary = filtered_df.groupBy("event_name").agg(
    count("*").alias("event_count"),
    sum("duration_ms").alias("total_duration")
)

# Step 4: Display results
print("User Summary (using cached data):")
display(user_summary)

print("\nEvent Summary (using cached data):")
display(event_summary)

# Step 5: Clean up cache
filtered_df.unpersist()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Query Optimization Best Practices
# MAGIC
# MAGIC **Quick Reference for Performance:**
# MAGIC
# MAGIC | Practice | Impact | Example |
# MAGIC |---|---|---|
# MAGIC | **Filter Early** | High | `.filter().agg()` not `.agg().filter()` |
# MAGIC | **Combine Filters** | High | Use AND, avoid multiple `.filter()` |
# MAGIC | **Cache Reused Data** | High | `.cache()` before multiple queries |
# MAGIC | **Use EXPLAIN** | Medium | Identify slow query plans |
# MAGIC | **Select Columns** | Medium | `.select()` reduces I/O |
# MAGIC | **Partition Data** | High | Distribute processing |

# COMMAND ----------

print("===== QUERY OPTIMIZATION BEST PRACTICES =====\n")

best_practices = {
    "1. Filter Early": "Remove unnecessary rows before aggregations",
    "2. Combine Filters": "Use single WHERE clause with AND, not multiple filters",
    "3. Cache Wisely": "Use .cache() only for reused DataFrames",
    "4. Check EXPLAIN": "Always review query plans for large queries",
    "5. Select Columns": "Use .select() to reduce data size",
    "6. Use Partitioning": "Distribute reads across partitions",
    "7. Avoid Shuffles": "Minimize wide transformations",
    "8. Unpersist Cache": "Release memory when done with cached data",
}

for practice, description in best_practices.items():
    print(f"{practice}:")
    print(f"  → {description}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Summary - Query Optimization
# MAGIC
# MAGIC **Key Concepts Covered:**

# COMMAND ----------

print("===== QUERY OPTIMIZATION SUMMARY =====")
print("✓ Understood query execution plans with EXPLAIN")
print("✓ Compared naive vs optimized filter approaches")
print("✓ Applied early filtering to reduce data size")
print("✓ Cached DataFrames for multiple queries")
print("✓ Unpersisted cache to free memory")
print("✓ Implemented complete optimization workflow")
print("✓ Learned best practices for performance")

print("\n========================================")
print("✓ QUERY OPTIMIZATION GUIDE COMPLETE!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook:
# MAGIC
# MAGIC 1. **Copy the content** into a Databricks Notebook
# MAGIC 2. **Run All Cells** to see optimization techniques
# MAGIC 3. **Review EXPLAIN outputs** to understand plans
# MAGIC 4. **Use patterns** in your own queries
# MAGIC
# MAGIC ## Performance Tips:
# MAGIC
# MAGIC - **Always FILTER EARLY** - biggest performance gain
# MAGIC - **CACHE before loops** - avoid recomputation
# MAGIC - **Use EXPLAIN** - understand before optimizing
# MAGIC - **Combine filters** - single WHERE > multiple .filter()
# MAGIC - **Monitor memory** - cache only necessary data
