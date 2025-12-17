# Databricks notebook source
# MAGIC %md
# MAGIC # Additional Useful Functions - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates practical PySpark functions for data transformation:
# MAGIC - **WHEN**: Conditional logic (IF-THEN)
# MAGIC - **COALESCE**: Handle NULL values
# MAGIC - **NULLIF**: Create NULL under conditions
# MAGIC - **NA Operations**: Handling missing data
# MAGIC - **JOINS**: Combining DataFrames

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, coalesce, nullif, lit, endswith, explode, count, na, drop, fillna, join
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType

print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create Sales DataFrame
# MAGIC
# MAGIC Creating sales data with:
# MAGIC - `transaction_id`: Transaction ID
# MAGIC - `email`: Customer email
# MAGIC - `total_amount`: Purchase amount
# MAGIC - `discount_percent`: Discount applied
# MAGIC - `items`: Products purchased

# COMMAND ----------

spark = SparkSession.builder.appName("AdditionalFunctions").getOrCreate()

# Sales Data
sales_data = [
    (1, "alice@gmail.com", 500.0, 10.0, ["Laptop", "Mouse"]),
    (2, "bob@yahoo.com", 150.0, 5.0, ["Keyboard"]),
    (3, "charlie@gmail.com", 1200.0, 15.0, ["Monitor", "Stand"]),
    (4, "david@outlook.com", 75.0, None, ["Cable"]),
    (5, "eve@gmail.com", 350.0, 0.0, ["Speaker", "Headphones"]),
    (6, "frank@yahoo.com", None, 10.0, ["Desk"]),
    (7, "grace@gmail.com", 200.0, 5.0, ["Chair"]),
    (8, "heidi@gmail.com", 450.0, None, ["Monitor", "Keyboard"]),
    (9, "ivan@outlook.com", 100.0, 0.0, ["USB Cable"]),
    (10, "judy@gmail.com", 600.0, 20.0, ["Laptop", "Case"]),
]

sales_schema = StructType([
    StructField("transaction_id", IntegerType(), True),
    StructField("email", StringType(), True),
    StructField("total_amount", DoubleType(), True),
    StructField("discount_percent", DoubleType(), True),
    StructField("items", ArrayType(StringType()), True),
])

sales_df = spark.createDataFrame(sales_data, sales_schema)

print("===== SALES DATA =====")
sales_df.printSchema()
display(sales_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: WHEN - Conditional Logic
# MAGIC
# MAGIC **Goal:** Create new columns based on conditions using `when()`.
# MAGIC
# MAGIC **Function:** `when(condition, value).otherwise(other_value)`

# COMMAND ----------

print("===== WHEN - CONDITIONAL LOGIC =====")

# Categorize customers by email domain
gmail_users = sales_df.filter(col("email").endswith("@gmail.com"))

print(f"Total customers: {sales_df.count()}")
print(f"Gmail customers: {gmail_users.count()}\n")

# Create customer segment using WHEN
segmented_df = sales_df.select(
    col("email"),
    col("total_amount"),
    when(col("email").endswith("@gmail.com"), "Gmail").alias("email_provider")
)

display(segmented_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: WHEN with Multiple Conditions
# MAGIC
# MAGIC **Goal:** Create complex conditional logic with multiple WHEN clauses.

# COMMAND ----------

print("===== WHEN WITH MULTIPLE CONDITIONS =====")

# Categorize by purchase amount and discount
category_df = sales_df.select(
    col("transaction_id"),
    col("email"),
    col("total_amount"),
    col("discount_percent"),
    when(col("total_amount") >= 500, "High Value")
        .when(col("total_amount") >= 200, "Medium Value")
        .when(col("total_amount") >= 1, "Low Value")
        .otherwise("No Purchase").alias("purchase_category"),
    when(col("discount_percent") >= 15, "Heavy Discount")
        .when(col("discount_percent") >= 5, "Standard Discount")
        .when(col("discount_percent") > 0, "Light Discount")
        .otherwise("No Discount").alias("discount_category")
)

display(category_df)

print("\nInterpretation:")
print("- Multiple WHEN conditions create tiered categorization")
print("- Useful for customer segmentation and targeting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: COALESCE - Handle NULL Values
# MAGIC
# MAGIC **Goal:** Replace NULL values with a default value using `coalesce()`.
# MAGIC
# MAGIC **Function:** `coalesce(col1, col2, ...)`
# MAGIC - Returns first non-NULL value from list

# COMMAND ----------

print("===== COALESCE - HANDLE NULLS =====")

# Fill discount_percent NULL with 0
# Fill total_amount NULL with 0
coalesced_df = sales_df.select(
    col("transaction_id"),
    col("email"),
    coalesce(col("total_amount"), lit(0)).alias("amount_filled"),
    coalesce(col("discount_percent"), lit(0)).alias("discount_filled")
)

display(coalesced_df)

print("\nInterpretation:")
print("- coalesce(total_amount, 0): Use amount if not NULL, else 0")
print("- Prevents NULL values in downstream calculations")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: NULLIF - Create NULL Conditionally
# MAGIC
# MAGIC **Goal:** Create NULL values based on conditions using `nullif()`.
# MAGIC
# MAGIC **Function:** `nullif(col, value)`
# MAGIC - Returns NULL if col equals value, else returns col

# COMMAND ----------

print("===== NULLIF - CREATE NULLS CONDITIONALLY =====")

# Set discount to NULL if it's 0 (no discount)
nullif_df = sales_df.select(
    col("transaction_id"),
    col("email"),
    col("discount_percent"),
    nullif(col("discount_percent"), 0).alias("discount_or_null")
)

display(nullif_df)

print("\nInterpretation:")
print("- nullif(discount, 0): Returns NULL when discount is 0")
print("- Useful to distinguish 'no discount' (NULL) from 'zero discount' (0)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: NA Operations - Handling Missing Data
# MAGIC
# MAGIC **Goal:** Handle NULL/NA values with built-in NA operations.
# MAGIC
# MAGIC **Functions:**
# MAGIC - `.na.drop()`: Remove rows with NULL
# MAGIC - `.na.fill()`: Replace NULL with values

# COMMAND ----------

print("===== NA OPERATIONS =====")

# Show original count
print(f"Total rows: {sales_df.count()}")

# Drop rows with ANY NULL
no_nulls = sales_df.na.drop()
print(f"Rows without any NULL: {no_nulls.count()}\n")

# Show dropped rows
print("Rows with NULL values:")
display(sales_df.filter(col("total_amount").isNull() | col("discount_percent").isNull()))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Fill NULL Values
# MAGIC
# MAGIC **Goal:** Fill NULL values with specific values using `.na.fill()`.

# COMMAND ----------

print("===== FILLING NULL VALUES =====")

# Fill NULL discount_percent with "NO DISCOUNT"
filled_df = sales_df.select(
    col("transaction_id"),
    col("email"),
    col("items"),
    explode(col("items")).alias("item"),
    col("discount_percent")
).na.fill({"discount_percent": 0})

print(f"Rows after filling NULLs: {filled_df.count()}\n")

display(filled_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Handling Missing Data in Arrays
# MAGIC
# MAGIC **Goal:** Count coupons (from array) and handle missing values.

# COMMAND ----------

print("===== ARRAY ANALYSIS WITH NULL HANDLING =====")

# Create coupon array
sales_with_coupons = sales_df.select(
    col("email"),
    when(col("discount_percent") > 0, ["SAVE10", "SAVE20"]).otherwise(None).alias("coupons")
)

# Explode and count
coupon_counts = sales_with_coupons.select(
    col("email"),
    explode(col("coupons")).alias("coupon")
)

print(f"Total rows with email/coupons: {sales_with_coupons.count()}")
print(f"Total coupon records: {coupon_counts.count()}\n")

display(coupon_counts.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: JOIN Operations
# MAGIC
# MAGIC **Goal:** Combine two DataFrames using various JOIN types.
# MAGIC
# MAGIC **JOIN Types:**
# MAGIC - `inner`: Only matching rows
# MAGIC - `left`: All from left + matching from right
# MAGIC - `right`: All from right + matching from left
# MAGIC - `outer`: All rows from both

# COMMAND ----------

print("===== JOIN OPERATIONS =====")

# Create user segmentation DataFrame
users_data = [
    ("alice@gmail.com", "Premium", "USA"),
    ("bob@yahoo.com", "Standard", "UK"),
    ("charlie@gmail.com", "Premium", "USA"),
    ("david@outlook.com", "Basic", "Canada"),
    ("eve@gmail.com", "Premium", "USA"),
]

users_df = spark.createDataFrame(users_data, ["email", "tier", "country"])

print("Users DataFrame:")
display(users_df)

# Inner join - only customers in both
joined_df = sales_df.join(users_df, on="email", how="inner")

print(f"\nSales: {sales_df.count()} rows")
print(f"Users: {users_df.count()} rows")
print(f"Inner Join: {joined_df.count()} rows\n")

display(joined_df.select("email", "total_amount", "tier", "country"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Different JOIN Types
# MAGIC
# MAGIC **Goal:** Compare different JOIN strategies.

# COMMAND ----------

print("===== COMPARING JOIN TYPES =====")

# Left join - all sales, with user info if available
left_join = sales_df.join(users_df, on="email", how="left")
print(f"Left Join: {left_join.count()} rows (all sales)")

# Outer join - all rows from both
outer_join = sales_df.join(users_df, on="email", how="outer")
print(f"Outer Join: {outer_join.count()} rows (all sales + all users)")

display(outer_join.select("email", "total_amount", "tier"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Practical Example - Customer Analysis
# MAGIC
# MAGIC **Goal:** Combine WHEN, COALESCE, and JOIN for complete analysis.

# COMMAND ----------

print("===== PRACTICAL EXAMPLE: CUSTOMER ANALYSIS =====")

# Combine all operations
customer_analysis = sales_df.join(users_df, on="email", how="left").select(
    col("email"),
    col("tier"),
    col("country"),
    coalesce(col("total_amount"), lit(0)).alias("purchase_amount"),
    coalesce(col("discount_percent"), lit(0)).alias("discount"),
    when(col("tier") == "Premium", "VIP").otherwise("Regular").alias("customer_type"),
    when(col("country") == "USA", "Domestic").otherwise("International").alias("region")
)

display(customer_analysis)

print("\nInterpretation:")
print("- Combines customer data with their purchase info")
print("- Handles missing values gracefully")
print("- Creates business-ready segments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 13: Summary - Additional Functions
# MAGIC
# MAGIC **Quick Reference:**
# MAGIC
# MAGIC | Function | Purpose | Example |
# MAGIC |---|---|---|
# MAGIC | `when()` | Conditional logic | `when(col("amount") > 100, "High")` |
# MAGIC | `coalesce()` | First non-NULL | `coalesce(col, lit(0))` |
# MAGIC | `nullif()` | Create NULLs | `nullif(col, 0)` |
# MAGIC | `.na.drop()` | Remove NULL rows | `df.na.drop()` |
# MAGIC | `.na.fill()` | Fill NULLs | `df.na.fill({"col": 0})` |
# MAGIC | `.join()` | Combine tables | `df1.join(df2, on="key")` |

# COMMAND ----------

print("===== ADDITIONAL FUNCTIONS SUMMARY =====")
print("✓ Created sales and user DataFrames")
print("✓ Used WHEN for conditional logic")
print("✓ Handled NULL values with COALESCE")
print("✓ Created NULLs with NULLIF")
print("✓ Dropped rows with NULLs")
print("✓ Filled missing data")
print("✓ Performed INNER and LEFT JOINs")
print("✓ Combined multiple operations")

print("\n========================================")
print("✓ ADDITIONAL FUNCTIONS GUIDE COMPLETE!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook:
# MAGIC
# MAGIC 1. **Copy the content** into a Databricks Notebook
# MAGIC 2. **Run All Cells** to see practical examples
# MAGIC 3. **Modify data** to explore different scenarios
# MAGIC
# MAGIC ## Key Takeaways:
# MAGIC
# MAGIC - **WHEN**: Use for any IF-THEN-ELSE logic
# MAGIC - **COALESCE**: First step in NULL handling
# MAGIC - **NULLIF**: Creates NULLs for conditions
# MAGIC - **NA Operations**: Built-in NULL management
# MAGIC - **JOIN**: Combine complementary data
