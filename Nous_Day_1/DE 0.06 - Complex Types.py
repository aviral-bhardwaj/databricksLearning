# Databricks notebook source
# MAGIC %md
# MAGIC # Complex Data Types - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates working with complex data types in PySpark:
# MAGIC - **STRUCT**: Nested columns (like objects)
# MAGIC - **ARRAY**: Collections of elements
# MAGIC - **MAP**: Key-value pairs
# MAGIC
# MAGIC Each section uses a practical e-commerce order DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, array, map_from_arrays, explode, split, element_at, array_contains, collect_set
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, MapType

print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create DataFrame with Complex Types
# MAGIC
# MAGIC Creating an e-commerce DataFrame with:
# MAGIC - `order_id`: Order ID
# MAGIC - `customer_email`: Customer email
# MAGIC - `items`: ARRAY of item names (ArrayType)
# MAGIC - `quantity`: ARRAY of quantities (ArrayType)
# MAGIC - `price`: ARRAY of prices (ArrayType)

# COMMAND ----------

# Initialize Spark Session
spark = SparkSession.builder.appName("ComplexTypes").getOrCreate()

# Sample E-commerce Order Data
data = [
    (101, "alice@example.com", ["Laptop", "Mouse"], [1, 2], [1200.00, 25.00]),
    (102, "bob@example.com", ["Mattress", "Pillow", "Sheet"], [1, 2, 3], [800.00, 50.00, 30.00]),
    (103, "charlie@example.com", ["Keyboard", "Monitor"], [1, 1], [120.00, 350.00]),
    (104, "david@example.com", ["Mattress", "Bed Frame"], [1, 1], [800.00, 400.00]),
    (105, "eve@example.com", ["Chair", "Desk", "Lamp"], [2, 1, 3], [250.00, 500.00, 75.00]),
    (106, "frank@example.com", ["Mattress"], [2], [800.00]),
    (107, "grace@example.com", ["Speaker", "Headphones", "Cable"], [1, 1, 2], [200.00, 150.00, 15.00]),
    (108, "heidi@example.com", ["Mattress", "Mattress Protector"], [1, 1], [800.00, 50.00]),
    (109, "ivan@example.com", ["Monitor"], [2], [350.00]),
    (110, "judy@example.com", ["Mattress", "Sheet", "Blanket"], [1, 4, 2], [800.00, 30.00, 60.00]),
]

# Define Schema with Complex Types
schema = StructType([
    StructField("order_id", IntegerType(), True),
    StructField("customer_email", StringType(), True),
    StructField("items", ArrayType(StringType()), True),
    StructField("quantity", ArrayType(IntegerType()), True),
    StructField("price", ArrayType(IntegerType()), True),
])

# Create DataFrame
df = spark.createDataFrame(data, schema=schema)

print("===== ORIGINAL DATAFRAME WITH COMPLEX TYPES =====")
df.printSchema()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Understanding Complex Data Types
# MAGIC
# MAGIC **ARRAY**: Collection of elements of the same type
# MAGIC - Example: `["Laptop", "Mouse", "Keyboard"]`
# MAGIC
# MAGIC **STRUCT**: Named collection of fields (like a Python dict)
# MAGIC - Example: `{name: "Laptop", price: 1200}`
# MAGIC
# MAGIC **MAP**: Key-value pairs
# MAGIC - Example: `{"Laptop": 1200, "Mouse": 25}`

# COMMAND ----------

print("===== COMPLEX DATA TYPES OVERVIEW =====\n")

print("1. ARRAY Type:")
print("   - Ordered collection of elements")
print("   - All elements have same type")
print("   - Example: ['item1', 'item2', 'item3']\n")

print("2. STRUCT Type:")
print("   - Named collection of fields")
print("   - Can contain different types")
print("   - Example: {name: 'Laptop', price: 1200, in_stock: true}\n")

print("3. MAP Type:")
print("   - Key-value pairs")
print("   - All keys same type, all values same type")
print("   - Example: {'Laptop': 1200, 'Mouse': 25}\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Exploding Arrays - Unnesting Data
# MAGIC
# MAGIC **Goal:** Convert array rows into multiple rows (one per element).
# MAGIC
# MAGIC **Function:** `explode(col)`
# MAGIC - Input: 1 row with array of N elements
# MAGIC - Output: N rows with 1 element each

# COMMAND ----------

print("===== EXPLODING ARRAYS =====")

# Explode items array - one row per item
exploded_df = df.select(
    col("order_id"),
    col("customer_email"),
    explode(col("items")).alias("item_name"),
    col("quantity"),
    col("price")
)

print("Before explode: 10 rows")
print(f"After explode: {exploded_df.count()} rows\n")

display(exploded_df.limit(15))

print("\nInterpretation:")
print("- Each item is now in a separate row")
print("- Original arrays are 'exploded' into multiple rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Checking Array Membership with array_contains()
# MAGIC
# MAGIC **Goal:** Check if an array contains a specific value.
# MAGIC
# MAGIC **Function:** `array_contains(col, value)`
# MAGIC - Returns TRUE/FALSE if value is in array

# COMMAND ----------

print("===== CHECKING ARRAY MEMBERSHIP =====")

contains_df = df.select(
    col("order_id"),
    col("customer_email"),
    col("items"),
    array_contains(col("items"), "Mattress").alias("has_mattress"),
    array_contains(col("items"), "Laptop").alias("has_laptop"),
    array_contains(col("items"), "Keyboard").alias("has_keyboard")
)

display(contains_df)

print("\nInterpretation:")
print("- has_mattress: TRUE if order contains Mattress")
print("- Useful for filtering orders with specific products")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Filtering Arrays with array_contains()
# MAGIC
# MAGIC **Goal:** Keep only rows where array contains specific value.

# COMMAND ----------

print("===== FILTERING BY ARRAY MEMBERSHIP =====")

mattress_orders_df = df.filter(array_contains(col("items"), "Mattress"))

print(f"Total orders: {df.count()}")
print(f"Orders with Mattress: {mattress_orders_df.count()}\n")

display(mattress_orders_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: String Operations - split()
# MAGIC
# MAGIC **Goal:** Split a string into an array.
# MAGIC
# MAGIC **Function:** `split(col, delimiter)`
# MAGIC - Input: String column
# MAGIC - Output: Array of substrings

# COMMAND ----------

print("===== STRING SPLIT INTO ARRAY =====")

# Split email into username and domain
email_split_df = df.select(
    col("customer_email"),
    split(col("customer_email"), "@").alias("email_parts"),
    element_at(split(col("customer_email"), "@"), 1).alias("username"),
    element_at(split(col("customer_email"), "@"), 2).alias("domain")
)

display(email_split_df)

print("\nInterpretation:")
print("- email_parts: Array after splitting by '@'")
print("- username: First part (before @)")
print("- domain: Second part (after @)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Creating STRUCT - Nested Columns
# MAGIC
# MAGIC **Goal:** Create structured/nested columns using `struct()`.
# MAGIC
# MAGIC **Function:** `struct(col1, col2, ...)`
# MAGIC - Combines multiple columns into a single STRUCT column

# COMMAND ----------

print("===== CREATING STRUCT COLUMNS =====")

# Create nested structure with order details
struct_df = df.select(
    col("order_id"),
    col("customer_email"),
    struct(
        col("items").alias("item_list"),
        col("quantity").alias("qty"),
        col("price").alias("prices")
    ).alias("order_details")
)

print("Schema with STRUCT:")
struct_df.printSchema()

display(struct_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Accessing STRUCT Fields with dot notation
# MAGIC
# MAGIC **Goal:** Access fields inside a STRUCT column.
# MAGIC
# MAGIC **Syntax:** `col("struct_col.field_name")`

# COMMAND ----------

print("===== ACCESSING STRUCT FIELDS =====")

# Access fields from nested structure
accessed_df = df.select(
    col("order_id"),
    struct(
        col("items"),
        col("quantity"),
        col("price")
    ).alias("order_details")
).select(
    col("order_id"),
    col("order_details.items").alias("items_from_struct"),
    col("order_details.quantity").alias("qty_from_struct")
)

display(accessed_df.limit(5))

print("\nInterpretation:")
print("- order_details.items: Access 'items' field from struct")
print("- order_details.quantity: Access 'quantity' field from struct")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Aggregating Arrays with collect_set()
# MAGIC
# MAGIC **Goal:** Collect unique values from an array column across multiple rows.
# MAGIC
# MAGIC **Function:** `collect_set(col)`
# MAGIC - Combines values from multiple rows into a single array
# MAGIC - Removes duplicates

# COMMAND ----------

print("===== AGGREGATING ARRAYS =====")

# Collect all unique items per customer
items_per_customer = df.select(
    col("customer_email"),
    explode(col("items")).alias("item")
).groupBy("customer_email").agg(
    collect_set(col("item")).alias("all_items")
)

display(items_per_customer)

print("\nInterpretation:")
print("- Groups by customer_email")
print("- Collects unique items per customer")
print("- Result is an ARRAY column")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Flattening Complex Types
# MAGIC
# MAGIC **Goal:** Convert complex nested structures into flat columns.

# COMMAND ----------

print("===== FLATTENING COMPLEX TYPES =====")

# Flatten by exploding all arrays together
flattened_df = df.select(
    col("order_id"),
    col("customer_email"),
    col("items"),
    col("quantity"),
    col("price")
).select(
    col("order_id"),
    col("customer_email"),
    explode(col("items")).alias("item_name"),
    explode(col("quantity")).alias("qty"),
    explode(col("price")).alias("price")
)

display(flattened_df.limit(10))

print("\nInterpretation:")
print("- Original arrays are exploded into separate rows")
print("- Each row represents one item with its quantity and price")
print("- Complex types become flat, simple columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 13: Advanced - Map Operations
# MAGIC
# MAGIC **Goal:** Work with MAP type (key-value pairs).

# COMMAND ----------

print("===== MAP TYPE OPERATIONS =====")

# Create a map from item names and prices
map_df = df.select(
    col("order_id"),
    col("customer_email"),
    map_from_arrays(col("items"), col("price")).alias("item_price_map")
)

print("Schema with MAP type:")
map_df.printSchema()

display(map_df.limit(5))

print("\nInterpretation:")
print("- item_price_map: MAP type with item names as keys, prices as values")
print("- Example: {'Laptop': 1200, 'Mouse': 25}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 14: Analyzing Mattress Orders
# MAGIC
# MAGIC **Goal:** Complex real-world analysis combining multiple operations.

# COMMAND ----------

print("===== MATTRESS ORDERS ANALYSIS =====")

# Filter orders with Mattress, extract details
mattress_analysis = df.filter(
    array_contains(col("items"), "Mattress")
).select(
    col("order_id"),
    col("customer_email"),
    element_at(split(col("customer_email"), "@"), 1).alias("username"),
    col("items"),
    explode(col("items")).alias("item_name"),
    element_at(col("items"), 1).alias("first_item")
).filter(
    col("item_name") == "Mattress"
)

display(mattress_analysis)

print(f"\nTotal Mattress orders: {mattress_analysis.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 15: Summary - Complex Types Operations
# MAGIC
# MAGIC **Quick Reference:**
# MAGIC
# MAGIC | Function | Purpose | Example |
# MAGIC |---|---|---|
# MAGIC | `explode` | Unnest array into rows | `explode(col("items"))` |
# MAGIC | `element_at` | Access array element | `element_at(col("items"), 1)` |
# MAGIC | `array_contains` | Check if item in array | `array_contains(col("items"), "Laptop")` |
# MAGIC | `split` | Split string to array | `split(col("email"), "@")` |
# MAGIC | `struct` | Create nested structure | `struct(col1, col2)` |
# MAGIC | `collect_set` | Aggregate into array | `collect_set(col("item"))` |
# MAGIC | `map_from_arrays` | Create key-value pairs | `map_from_arrays(keys, values)` |

# COMMAND ----------

print("===== COMPLEX TYPES OPERATIONS SUMMARY =====")
print("✓ Created DataFrame with ARRAY types")
print("✓ Exploded arrays into multiple rows")
print("✓ Accessed array elements by position")
print("✓ Checked array membership")
print("✓ Filtered based on array content")
print("✓ Split strings into arrays")
print("✓ Created STRUCT (nested) columns")
print("✓ Accessed STRUCT fields with dot notation")
print("✓ Aggregated arrays with collect_set")
print("✓ Flattened complex types")
print("✓ Created and worked with MAP types")

print("\n========================================")
print("✓ COMPLEX DATA TYPES GUIDE COMPLETE!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook:
# MAGIC
# MAGIC 1. **Copy the content** into a Databricks Notebook
# MAGIC 2. **Run All Cells** to see all complex type operations
# MAGIC 3. **Experiment** by modifying the sample data
# MAGIC
# MAGIC ## Key Takeaways:
# MAGIC
# MAGIC - **ARRAY**: Use `explode()` to unnest, `element_at()` to access
# MAGIC - **STRUCT**: Use dot notation `struct.field` to access nested data
# MAGIC - **MAP**: Use for key-value associations
# MAGIC - **Flattening**: `explode()` converts complex to simple columns
# MAGIC - **Filtering**: Use `array_contains()` for membership checks
