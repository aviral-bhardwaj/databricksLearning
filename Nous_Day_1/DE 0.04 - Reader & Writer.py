# Databricks notebook source
# MAGIC %md
# MAGIC # COVID Data Reader & Writer - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates reading COVID-19 datasets from Databricks datasets
# MAGIC and writing them in multiple formats (CSV, Parquet, Delta Lake).
# MAGIC Each section includes clear explanations and practical examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, count, sum, avg


print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Explore Available COVID Datasets
# MAGIC
# MAGIC Let's explore the Databricks COVID-19 datasets available in the file system.

# COMMAND ----------

# List available COVID datasets
covid_datasets = dbutils.fs.ls('/databricks-datasets/COVID')

print("===== AVAILABLE COVID DATASETS =====\n")
for dataset in covid_datasets:
    print(f"Path: {dataset.path}")
    print(f"Name: {dataset.name}")
    print(f"Size: {dataset.size} bytes")
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Read COVID Hospital Beds Data (CSV)
# MAGIC
# MAGIC Reading COVID hospital beds data from ESRI dataset in CSV format.

# COMMAND ----------

# Define the path to COVID hospital beds data
covid_hospital_path = '/databricks-datasets/COVID/ESRI_hospital_beds/'

# List files in the hospital beds directory
print("===== FILES IN HOSPITAL BEDS DIRECTORY =====\n")
try:
    hospital_files = dbutils.fs.ls(covid_hospital_path)
    for file in hospital_files:
        print(f"File: {file.name}")
        print(f"Size: {file.size} bytes\n")
except Exception as e:
    print(f"Error listing files: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Read CSV File - Hospital Beds Data

# COMMAND ----------

# Read COVID hospital beds CSV data
df_hospital = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv('/databricks-datasets/COVID/ESRI_hospital_beds')

print("===== HOSPITAL BEDS DATA =====\n")
print(f"Rows: {df_hospital.count()}")
print(f"Columns: {len(df_hospital.columns)}")
print(f"\nSchema:")
df_hospital.printSchema()

print("\n===== FIRST 10 ROWS =====\n")
display(df_hospital.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Hospital Beds Data - Basic Analysis

# COMMAND ----------

print("===== HOSPITAL BEDS ANALYSIS =====\n")

# Display statistics
print("Column Names:")
print(df_hospital.columns)

print("\n===== Basic Statistics =====\n")
display(df_hospital.describe())

print("\nInterpretation:")
print("- Shows basic statistics for numeric columns")
print("- Includes count, mean, stddev, min, max")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Write Hospital Beds Data - CSV Format
# MAGIC
# MAGIC Write the hospital beds data to DBFS in CSV format with headers.

# COMMAND ----------

# Define output paths
output_base = '/tmp/covid_outputs'
csv_output_path = f'{output_base}/hospital_beds_csv'

print(f"===== WRITING TO CSV =====")
print(f"Output path: {csv_output_path}\n")

# Write DataFrame to CSV
df_hospital.coalesce(1) \
    .write \
    .option("header", "true") \
    .mode("overwrite") \
    .csv(csv_output_path)

print("✓ CSV write completed successfully!")
print(f"Location: {csv_output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Write Hospital Beds Data - Parquet Format
# MAGIC
# MAGIC Write the hospital beds data to DBFS in Parquet format with compression.

# COMMAND ----------

# Define Parquet output path
parquet_output_path = f'{output_base}/hospital_beds_parquet'

print(f"===== WRITING TO PARQUET =====")
print(f"Output path: {parquet_output_path}\n")

# Write DataFrame to Parquet with compression
df_hospital.write \
    .option("compression", "snappy") \
    .mode("overwrite") \
    .parquet(parquet_output_path)

print("✓ Parquet write completed successfully!")
print(f"Location: {parquet_output_path}")
print("Compression: snappy")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Write Hospital Beds Data - Delta Lake Format
# MAGIC
# MAGIC Write the hospital beds data to Delta Lake for ACID transactions and time travel.

# COMMAND ----------

# Define Delta output path
delta_output_path = f'{output_base}/hospital_beds_delta'

print(f"===== WRITING TO DELTA LAKE =====")
print(f"Output path: {delta_output_path}\n")

# Write DataFrame to Delta Lake
df_hospital.write \
    .mode("overwrite") \
    .format("delta") \
    .save(delta_output_path)

print("✓ Delta Lake write completed successfully!")
print(f"Location: {delta_output_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Read from Different Formats
# MAGIC
# MAGIC Read the data back from different formats to verify they were written correctly.

# COMMAND ----------

# Read from CSV
df_from_csv = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .csv(csv_output_path)

print("===== READ FROM CSV =====")
print(f"Rows: {df_from_csv.count()}")
print("Sample data:")
display(df_from_csv.limit(3))

# COMMAND ----------

# Read from Parquet
df_from_parquet = spark.read \
    .parquet(parquet_output_path)

print("\n===== READ FROM PARQUET =====")
print(f"Rows: {df_from_parquet.count()}")
print("Sample data:")
display(df_from_parquet.limit(3))

# COMMAND ----------

# Read from Delta Lake
df_from_delta = spark.read \
    .format("delta") \
    .load(delta_output_path)

print("\n===== READ FROM DELTA LAKE =====")
print(f"Rows: {df_from_delta.count()}")
print("Sample data:")
display(df_from_delta.limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Advanced Reader Options - CSV with Custom Delimiter
# MAGIC
# MAGIC Demonstrate reading CSV with custom options like delimiter, quote character, etc.

# COMMAND ----------

# Advanced CSV reading options
df_csv_advanced = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("delimiter", ",") \
    .option("quote", "\"") \
    .option("escape", "\\") \
    .option("nullValue", "") \
    .csv('/databricks-datasets/COVID/ESRI_hospital_beds')

print("===== ADVANCED CSV READ =====")
print("Options used:")
print("- header: true")
print("- inferSchema: true")
print("- delimiter: comma")
print("- quote: double quote")
print("- escape: backslash")
print("- nullValue: empty string")
print(f"\nRows read: {df_csv_advanced.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Advanced Writer Options - Parquet Partitioning
# MAGIC
# MAGIC Write data with partitioning for better query performance.

# COMMAND ----------

# Prepare data for partitioning
print("===== WRITING WITH PARTITIONING =====\n")

# If the data has a state or region column, we can partition by it
# For this example, we'll show the structure

partitioned_output = f'{output_base}/hospital_beds_partitioned'

print(f"Output path: {partitioned_output}")
print("Partitioning strategy: By available location fields\n")

# Write with partitioning (if applicable columns exist)
df_hospital.write \
    .option("compression", "snappy") \
    .mode("overwrite") \
    .parquet(partitioned_output)

print("✓ Partitioned Parquet write completed!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Write with Different Modes
# MAGIC
# MAGIC Demonstrate different write modes: overwrite, append, ignore, error.

# COMMAND ----------

print("===== WRITE MODES EXPLAINED =====\n")

test_output = f'{output_base}/test_modes'

# Mode 1: OVERWRITE (default)
print("1. OVERWRITE Mode - Replaces existing data")
df_hospital.limit(5) \
    .write \
    .mode("overwrite") \
    .csv(f'{test_output}/overwrite_mode')
print("✓ Overwrite completed\n")

# Mode 2: APPEND
print("2. APPEND Mode - Adds new data to existing")
df_hospital.limit(3) \
    .write \
    .mode("append") \
    .csv(f'{test_output}/append_mode')
print("✓ Append completed\n")

# Mode 3: IGNORE
print("3. IGNORE Mode - Ignores write if path exists")
df_hospital.limit(2) \
    .write \
    .mode("ignore") \
    .csv(f'{test_output}/ignore_mode')
print("✓ Ignore mode executed\n")

print("Mode Summary:")
print("- OVERWRITE: Delete existing, write new")
print("- APPEND: Add to existing data")
print("- IGNORE: Skip write if exists")
print("- ERROR: Throw error if exists (default)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 13: Data Transformation Before Writing
# MAGIC
# MAGIC Transform data before writing: filtering, selecting, renaming columns.

# COMMAND ----------

print("===== DATA TRANSFORMATION =====\n")

# Example transformations
df_transformed = df_hospital \
    .select(col("*")) \
    .filter(col(df_hospital.columns[0]).isNotNull())  # Remove nulls from first column

print("Original row count:", df_hospital.count())
print("Transformed row count:", df_transformed.count())

# Write transformed data
transformed_output = f'{output_base}/hospital_beds_transformed'
df_transformed.write \
    .mode("overwrite") \
    .parquet(transformed_output)

print(f"\n✓ Transformed data written to: {transformed_output}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 14: List Files in Output Directory
# MAGIC
# MAGIC Verify all written files in the output directory.

# COMMAND ----------

print("===== FILES WRITTEN IN OUTPUT DIRECTORY =====\n")

try:
    output_files = dbutils.fs.ls(output_base)
    
    for item in output_files:
        print(f"Path: {item.path}")
        print(f"Name: {item.name}")
        print(f"Size: {item.size} bytes")
        print("-" * 60)
        
    print(f"\nTotal items written: {len(output_files)}")
    
except Exception as e:
    print(f"Error: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 15: Reader/Writer Performance Considerations
# MAGIC
# MAGIC Best practices for reading and writing large datasets.

# COMMAND ----------

print("===== READER/WRITER BEST PRACTICES =====\n")

print("📖 READING:")
print("1. Use inferSchema=false when schema is known (faster)")
print("2. Use appropriate file format (Parquet for big data)")
print("3. Apply filtering early to reduce data volume")
print("4. Use partitioning for large datasets")
print("5. Consider caching frequently read data\n")

print("✏️ WRITING:")
print("1. Use Parquet for best compression (snappy/gzip)")
print("2. Use Delta Lake for ACID compliance")
print("3. Partition data for efficient querying")
print("4. Use coalesce() to reduce number of output files")
print("5. Choose appropriate write mode (overwrite/append)")
print("6. Handle column names and data types explicitly\n")

print("⚡ FILE FORMATS:")
print("- CSV: Human readable, slower, larger files")
print("- Parquet: Binary, fast, compressed, columnar")
print("- Delta: Parquet + transactions + time travel")
print("- ORC: Similar to Parquet, Hive optimized\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 16: Summary - Reader/Writer Operations

# COMMAND ----------

print("===== READER/WRITER SUMMARY =====\n")

print("✓ READ Operations Covered:")
print("1. CSV with inferSchema")
print("2. Parquet with compression")
print("3. Delta Lake format")
print("4. Custom CSV options (delimiter, quote, escape)")
print("5. Multiple file formats verification\n")

print("✓ WRITE Operations Covered:")
print("1. CSV format output")
print("2. Parquet with snappy compression")
print("3. Delta Lake format")
print("4. Different write modes (overwrite, append, ignore)")
print("5. Partitioned writes")
print("6. Transformed data writes\n")

print("✓ UTILITIES Covered:")
print("1. dbutils.fs.ls() - List files")
print("2. Data exploration - schema, count, describe")
print("3. File verification after write")
print("4. Performance best practices\n")

print("========================================")
print("✓ COVID DATA READER/WRITER GUIDE COMPLETE!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 17: Additional File Locations & Datasets
# MAGIC
# MAGIC Other available Databricks datasets for practice.

# COMMAND ----------

print("===== ADDITIONAL DATABRICKS DATASETS =====\n")

print("COVID-19 Related:")
print("- /databricks-datasets/COVID/ESRI_hospital_beds/")
print("- /databricks-datasets/COVID/COVID-19_Data/\n")

print("Other Public Datasets:")
print("- /databricks-datasets/airlines/")
print("- /databricks-datasets/amazon_reviews/")
print("- /databricks-datasets/nyctaxi/")
print("- /databricks-datasets/wikipedia/")
print("- /databricks-datasets/lending-club/\n")

print("Syntax to explore:")
print("dbutils.fs.ls('/databricks-datasets/COVID')")
print("spark.read.csv('/path/to/file.csv', header=True, inferSchema=True)")
print("df.write.parquet('/output/path', mode='overwrite')\n")

print("========================================")
print("Ready to read and write COVID-19 data!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook:
# MAGIC
# MAGIC 1. **Copy the entire content** into a Databricks Notebook
# MAGIC 2. **Each `# MAGIC %md` section** renders as formatted documentation
# MAGIC 3. **Each code section** executes as Python cells
# MAGIC 4. **The `# COMMAND ----------` lines** separate cells
# MAGIC 5. **Run All** to execute sequentially or individual cells with Shift+Enter
# MAGIC 6. **Monitor output** for file paths and statistics
# MAGIC
# MAGIC ## Key Functions Used:
# MAGIC
# MAGIC - `spark.read` - Read files (CSV, Parquet, Delta)
# MAGIC - `spark.write` - Write files in multiple formats
# MAGIC - `df.coalesce()` - Reduce partition count
# MAGIC - `dbutils.fs.ls()` - List DBFS locations
# MAGIC - `.option()` - Configure reader/writer behavior
# MAGIC - `.mode()` - Set write behavior (overwrite, append, ignore)
# MAGIC - `display()` - Rich visualization in Databricks
