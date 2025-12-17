# Databricks notebook source
# MAGIC %md
# MAGIC # Abandoned Carts Analysis - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates analyzing abandoned shopping carts using PySpark.
# MAGIC We'll identify customers who added items but didn't complete checkout.
# MAGIC Each section includes clear explanations and practical analysis examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum, avg, max, min, datediff, current_date, when, explode, collect_set, struct, to_date
from datetime import datetime, timedelta

print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create Sample Data - Customer Events
# MAGIC
# MAGIC Creating event data with:
# MAGIC - `event_id`: Event ID
# MAGIC - `customer_id`: Customer ID
# MAGIC - `customer_email`: Email address
# MAGIC - `event_type`: Type of event (cart_add, checkout_start, purchase, abandoned)
# MAGIC - `event_date`: When event occurred
# MAGIC - `items`: Products in cart
# MAGIC - `total_value`: Cart total

# COMMAND ----------

spark = SparkSession.builder.appName("AbandonedCartsAnalysis").getOrCreate()

# Customer Events Data
events_data = [
    (1, 101, "alice@example.com", "cart_add", "2023-12-01", ["Laptop"], 1200.00),
    (2, 101, "alice@example.com", "checkout_start", "2023-12-01", ["Laptop"], 1200.00),
    (3, 101, "alice@example.com", "purchase", "2023-12-01", ["Laptop"], 1200.00),
    
    (4, 102, "bob@example.com", "cart_add", "2023-12-02", ["Mattress"], 800.00),
    (5, 102, "bob@example.com", "abandoned", "2023-12-02", ["Mattress"], 800.00),
    
    (6, 103, "charlie@example.com", "cart_add", "2023-12-03", ["Keyboard", "Monitor"], 470.00),
    (7, 103, "charlie@example.com", "checkout_start", "2023-12-03", ["Keyboard", "Monitor"], 470.00),
    (8, 103, "charlie@example.com", "abandoned", "2023-12-03", ["Keyboard", "Monitor"], 470.00),
    
    (9, 104, "david@example.com", "cart_add", "2023-12-04", ["Chair"], 250.00),
    (10, 104, "david@example.com", "abandoned", "2023-12-04", ["Chair"], 250.00),
    
    (11, 105, "eve@example.com", "cart_add", "2023-12-05", ["Desk"], 500.00),
    (12, 105, "eve@example.com", "checkout_start", "2023-12-05", ["Desk"], 500.00),
    (13, 105, "eve@example.com", "purchase", "2023-12-05", ["Desk"], 500.00),
    
    (14, 106, "frank@example.com", "cart_add", "2023-12-06", ["Speaker", "Headphones"], 350.00),
    (15, 106, "frank@example.com", "abandoned", "2023-12-06", ["Speaker", "Headphones"], 350.00),
    
    (16, 107, "grace@example.com", "cart_add", "2023-12-07", ["Monitor"], 350.00),
    (17, 107, "grace@example.com", "checkout_start", "2023-12-07", ["Monitor"], 350.00),
    (18, 107, "grace@example.com", "purchase", "2023-12-07", ["Monitor"], 350.00),
    
    (19, 108, "heidi@example.com", "cart_add", "2023-12-08", ["Lamp", "Desk Organizer"], 125.00),
    (20, 108, "heidi@example.com", "abandoned", "2023-12-08", ["Lamp", "Desk Organizer"], 125.00),
]

# Define Schema
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType, ArrayType

schema = StructType([
    StructField("event_id", IntegerType(), True),
    StructField("customer_id", IntegerType(), True),
    StructField("customer_email", StringType(), True),
    StructField("event_type", StringType(), True),
    StructField("event_date", StringType(), True),
    StructField("items", ArrayType(StringType()), True),
    StructField("total_value", DoubleType(), True),
])

# Create DataFrame
events_df = spark.createDataFrame(events_data, schema=schema)

print("===== CUSTOMER EVENTS DATA =====")
events_df.printSchema()
display(events_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Identify Abandoned Carts
# MAGIC
# MAGIC **Goal:** Find customers who added items to cart but abandoned without purchasing.

# COMMAND ----------

print("===== ABANDONED CARTS IDENTIFICATION =====\n")

# Filter for abandoned events
abandoned_carts = events_df.filter(col("event_type") == "abandoned")

print(f"Total events: {events_df.count()}")
print(f"Abandoned cart events: {abandoned_carts.count()}\n")

display(abandoned_carts)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Abandoned Cart Statistics
# MAGIC
# MAGIC **Goal:** Calculate metrics for abandoned carts.

# COMMAND ----------

print("===== ABANDONED CARTS STATISTICS =====\n")

# Aggregate abandoned carts statistics
abandoned_stats = abandoned_carts.select(
    count("*").alias("total_abandoned"),
    sum("total_value").alias("total_lost_revenue"),
    avg("total_value").alias("avg_cart_value"),
    min("total_value").alias("min_cart_value"),
    max("total_value").alias("max_cart_value")
)

display(abandoned_stats)

print("\nInterpretation:")
print("- total_abandoned: Number of abandoned carts")
print("- total_lost_revenue: Sum of cart values not purchased")
print("- avg_cart_value: Average value of abandoned carts")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Abandoned Carts by Customer
# MAGIC
# MAGIC **Goal:** Aggregate abandoned carts per customer.

# COMMAND ----------

print("===== ABANDONED CARTS BY CUSTOMER =====\n")

abandoned_by_customer = abandoned_carts.groupBy("customer_id", "customer_email").agg(
    count("*").alias("abandonment_count"),
    sum("total_value").alias("lost_revenue"),
    collect_set("items").alias("abandoned_items"),
    collect_set("event_date").alias("abandonment_dates")
).orderBy(col("lost_revenue").desc())

display(abandoned_by_customer)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Customers with Multiple Abandonments
# MAGIC
# MAGIC **Goal:** Identify repeat offenders (customers with multiple abandoned carts).

# COMMAND ----------

print("===== REPEAT ABANDONERS =====\n")

repeat_abandoners = abandoned_by_customer.filter(col("abandonment_count") > 1)

print(f"Customers with 1+ abandoned carts: {abandoned_by_customer.count()}")
print(f"Customers with 2+ abandoned carts: {repeat_abandoners.count()}\n")

display(repeat_abandoners)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Purchase vs Abandonment Analysis
# MAGIC
# MAGIC **Goal:** Compare purchasing customers vs abandoning customers.

# COMMAND ----------

print("===== PURCHASE VS ABANDONMENT ANALYSIS =====\n")

# Customers who purchased
purchased_customers = events_df.filter(col("event_type") == "purchase").select("customer_id").distinct()

# Customers who abandoned
abandoned_customers = events_df.filter(col("event_type") == "abandoned").select("customer_id").distinct()

print(f"Customers who purchased: {purchased_customers.count()}")
print(f"Customers who abandoned: {abandoned_customers.count()}\n")

# Customers who did both
both = purchased_customers.join(abandoned_customers, "customer_id", "inner")
print(f"Customers who both purchased and abandoned: {both.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: High-Value Abandoned Carts
# MAGIC
# MAGIC **Goal:** Focus on high-value carts that were abandoned.

# COMMAND ----------

print("===== HIGH-VALUE ABANDONED CARTS =====\n")

# Abandoned carts over $400
high_value_abandoned = abandoned_carts.filter(col("total_value") > 400)

print(f"Abandoned carts > $400: {high_value_abandoned.count()}\n")

display(high_value_abandoned.select("customer_email", "items", "total_value", "event_date"))

print(f"Total lost revenue (>$400 carts): ${high_value_abandoned.agg(sum('total_value')).collect()[0][0]:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Products in Abandoned Carts
# MAGIC
# MAGIC **Goal:** Identify which products are frequently abandoned.

# COMMAND ----------

print("===== PRODUCTS IN ABANDONED CARTS =====\n")

# Explode items and count
product_abandonments = abandoned_carts.select(explode(col("items")).alias("product")).groupBy("product").agg(
    count("*").alias("abandonment_frequency")
).orderBy(col("abandonment_frequency").desc())

display(product_abandonments)

print("\nInterpretation:")
print("- Shows which products are most frequently abandoned")
print("- Useful for improving product descriptions or pricing")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Abandonment Rate by Customer
# MAGIC
# MAGIC **Goal:** Calculate what percentage of each customer's carts are abandoned.

# COMMAND ----------

print("===== ABANDONMENT RATE BY CUSTOMER =====\n")

# Get all customer events
customer_activity = events_df.groupBy("customer_id", "customer_email").agg(
    count(when(col("event_type") == "abandoned", 1)).alias("abandoned_count"),
    count("*").alias("total_events")
)

# Calculate abandonment rate
abandonment_rate = customer_activity.select(
    col("customer_id"),
    col("customer_email"),
    col("abandoned_count"),
    col("total_events"),
    (col("abandoned_count") / col("total_events") * 100).alias("abandonment_rate_pct")
).orderBy(col("abandonment_rate_pct").desc())

display(abandonment_rate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Summary - Abandoned Carts Analysis
# MAGIC
# MAGIC **Key Metrics:**
# MAGIC - Total abandoned carts
# MAGIC - Lost revenue
# MAGIC - Average abandoned cart value
# MAGIC - Repeat abandoners (retention risk)
# MAGIC - High-value abandoned carts

# COMMAND ----------

print("===== ABANDONED CARTS SUMMARY =====")
print("✓ Identified abandoned cart events")
print("✓ Calculated abandonment statistics")
print("✓ Aggregated by customer")
print("✓ Identified repeat abandoners")
print("✓ Compared purchase vs abandonment")
print("✓ Focused on high-value abandonments")
print("✓ Analyzed products in abandoned carts")
print("✓ Calculated abandonment rates")

print("\n========================================")
print("✓ ABANDONED CARTS ANALYSIS COMPLETE!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook:
# MAGIC
# MAGIC 1. **Copy the content** into a Databricks Notebook
# MAGIC 2. **Run All Cells** to see the complete analysis
# MAGIC 3. **Explore Results** for actionable insights
# MAGIC
# MAGIC ## Business Recommendations:
# MAGIC
# MAGIC - **Target high-value abandoners** with recovery emails
# MAGIC - **Analyze products** frequently abandoned
# MAGIC - **Reach out to repeat abandoners** with incentives
# MAGIC - **Optimize checkout** for users who abandon early
