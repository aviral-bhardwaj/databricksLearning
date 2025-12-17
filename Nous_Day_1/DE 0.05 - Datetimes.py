# Databricks notebook source
# MAGIC %md
# MAGIC # Date and Time Manipulation - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates comprehensive date and time operations using PySpark.
# MAGIC Each section includes clear explanations and practical examples using a custom employee DataFrame.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp, date_format, year, month, dayofweek, minute, second, date_add, datediff, current_date, current_timestamp, add_months
from pyspark.sql.types import TimestampType, DateType

print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create Employee DataFrame with Timestamps
# MAGIC
# MAGIC Creating a DataFrame with:
# MAGIC - `emp_id`: Employee ID
# MAGIC - `name`: Employee Name
# MAGIC - `join_date`: String date (YYYY-MM-DD)
# MAGIC - `event_timestamp`: Unix timestamp (microseconds)
# MAGIC - `login_time`: String timestamp

# COMMAND ----------

# Initialize Spark Session
spark = SparkSession.builder.appName("DateTimeOperations").getOrCreate()

# Sample Data: Employee Login Events
data = [
    (1, "Alice",   "2023-01-15", 1673769600000000, "2023-01-15 08:30:00"),
    (2, "Bob",     "2022-05-20", 1653033600000000, "2022-05-20 09:15:00"),
    (3, "Charlie", "2023-03-10", 1678435200000000, "2023-03-10 08:45:00"),
    (4, "David",   "2021-11-05", 1636108800000000, "2021-11-05 09:00:00"),
    (5, "Eve",     "2023-07-25", 1690272000000000, "2023-07-25 08:55:00"),
    (6, "Frank",   "2022-08-12", 1660291200000000, "2022-08-12 09:30:00"),
    (7, "Grace",   "2023-02-28", 1677571200000000, "2023-02-28 08:40:00"),
    (8, "Heidi",   "2021-06-18", 1623993600000000, "2021-06-18 09:10:00"),
    (9, "Ivan",    "2023-09-05", 1693891200000000, "2023-09-05 08:50:00"),
    (10,"Judy",    "2022-12-01", 1669881600000000, "2022-12-01 09:05:00"),
    (11,"Ken",     "2023-04-15", 1681545600000000, "2023-04-15 08:35:00"),
    (12,"Leo",     "2021-09-30", 1632988800000000, "2021-09-30 09:25:00"),
    (13,"Mallory", "2023-06-20", 1687248000000000, "2023-06-20 08:58:00"),
    (14,"Niaj",    "2022-03-14", 1647244800000000, "2022-03-14 09:20:00"),
    (15,"Olivia",  "2023-08-08", 1691481600000000, "2023-08-08 08:42:00"),
    (16,"Peggy",   "2021-10-22", 1634889600000000, "2021-10-22 09:12:00"),
    (17,"Quentin", "2023-05-18", 1684396800000000, "2023-05-18 08:48:00"),
    (18,"Rupert",  "2022-07-07", 1657180800000000, "2022-07-07 09:22:00"),
    (19,"Sybil",   "2023-10-10", 1696924800000000, "2023-10-10 08:52:00"),
    (20,"Trent",   "2021-12-25", 1640419200000000, "2021-12-25 09:18:00"),
]

# Define Schema
schema = "emp_id INT, name STRING, join_date_str STRING, event_timestamp_long LONG, login_time_str STRING"

# Create DataFrame
df = spark.createDataFrame(data, schema=schema)

print("===== ORIGINAL DATA FRAME =====")
df.printSchema()
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Convert Numeric Timestamp to Timestamp Type
# MAGIC
# MAGIC **Goal:** Convert Unix timestamp (microseconds) to proper TimestampType.
# MAGIC
# MAGIC **Logic:**
# MAGIC - Divide by 1e6 (1,000,000) to convert microseconds to seconds
# MAGIC - Cast to `TimestampType` or use `to_timestamp`

# COMMAND ----------

print("===== CONVERTING NUMERIC TIMESTAMP =====")

# Convert microseconds to seconds and cast to timestamp
timestamp_df = df.withColumn(
    "event_timestamp", 
    (col("event_timestamp_long") / 1e6).cast(TimestampType())
)

display(timestamp_df.select("emp_id", "name", "event_timestamp_long", "event_timestamp"))

print("\nInterpretation:")
print("- `event_timestamp_long`: Raw numeric value")
print("- `event_timestamp`: Converted datetime object")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Formatting Dates and Times
# MAGIC
# MAGIC **Goal:** Format timestamp into specific string patterns using `date_format`.
# MAGIC
# MAGIC **Patterns:**
# MAGIC - `MMMM dd, yyyy` -> Full month name, day, year (e.g., January 15, 2023)
# MAGIC - `HH:mm:ss` -> 24-hour time format
# MAGIC - `MM-dd-yyyy` -> US date format

# COMMAND ----------

print("===== FORMATTING DATE AND TIME =====")

formatted_df = timestamp_df \
    .withColumn("date_string", date_format("event_timestamp", "MMMM dd, yyyy")) \
    .withColumn("time_string", date_format("event_timestamp", "HH:mm:ss")) \
    .withColumn("us_date_format", date_format("event_timestamp", "MM-dd-yyyy"))

display(formatted_df.select("emp_id", "name", "event_timestamp", "date_string", "time_string", "us_date_format"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: Extracting Date Components
# MAGIC
# MAGIC **Goal:** Extract specific parts of the timestamp (Year, Month, Day, etc.).
# MAGIC
# MAGIC **Functions:**
# MAGIC - `year()`: Extract year
# MAGIC - `month()`: Extract month number (1-12)
# MAGIC - `dayofweek()`: Extract day of week (1=Sunday, 7=Saturday)
# MAGIC - `minute()`, `second()`: Extract time components

# COMMAND ----------

print("===== EXTRACTING DATE COMPONENTS =====")

datetime_components_df = timestamp_df \
    .withColumn("year", year(col("event_timestamp"))) \
    .withColumn("month", month(col("event_timestamp"))) \
    .withColumn("day_of_week", dayofweek(col("event_timestamp"))) \
    .withColumn("minute", minute(col("event_timestamp"))) \
    .withColumn("second", second(col("event_timestamp")))

display(datetime_components_df.select("emp_id", "event_timestamp", "year", "month", "day_of_week", "minute"))

print("\nInterpretation:")
print("- day_of_week: 1 = Sunday, 2 = Monday, ..., 7 = Saturday")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: Convert Timestamp to Date
# MAGIC
# MAGIC **Goal:** Extract only the date part (removing time) using `to_date()`.
# MAGIC
# MAGIC **Use Case:** When analysis requires daily granularity, ignoring specific times.

# COMMAND ----------

print("===== CONVERTING TIMESTAMP TO DATE =====")

date_df = timestamp_df.withColumn("date_only", to_date(col("event_timestamp")))

display(date_df.select("emp_id", "event_timestamp", "date_only"))

print("\nInterpretation:")
print("- `event_timestamp`: Contains date and time")
print("- `date_only`: Contains only date (YYYY-MM-DD)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: Date Arithmetic - Adding Days
# MAGIC
# MAGIC **Goal:** Add or subtract days from a date using `date_add()`.
# MAGIC
# MAGIC **Functions:**
# MAGIC - `date_add(col, days)`: Adds `days` to the date column.
# MAGIC - Negative values subtract days.

# COMMAND ----------

print("===== DATE ARITHMETIC (ADDING DAYS) =====")

plus_days_df = date_df \
    .withColumn("plus_2_days", date_add(col("date_only"), 2)) \
    .withColumn("minus_7_days", date_add(col("date_only"), -7))

display(plus_days_df.select("emp_id", "date_only", "plus_2_days", "minus_7_days"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: Date Arithmetic - Adding Months
# MAGIC
# MAGIC **Goal:** Add months to a date using `add_months()`.

# COMMAND ----------

print("===== DATE ARITHMETIC (ADDING MONTHS) =====")

plus_months_df = date_df \
    .withColumn("next_month", add_months(col("date_only"), 1)) \
    .withColumn("last_quarter", add_months(col("date_only"), -3))

display(plus_months_df.select("emp_id", "date_only", "next_month", "last_quarter"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: Calculating Date Differences
# MAGIC
# MAGIC **Goal:** Calculate the number of days between two dates using `datediff()`.
# MAGIC
# MAGIC **Function:**
# MAGIC - `datediff(end_date, start_date)`: Returns `end_date - start_date` in days.

# COMMAND ----------

print("===== DATE DIFFERENCES =====")

# Calculate days since joining until current date
diff_df = date_df \
    .withColumn("current_date", current_date()) \
    .withColumn("days_since_event", datediff(current_date(), col("date_only")))

display(diff_df.select("emp_id", "date_only", "current_date", "days_since_event"))

print("\nInterpretation:")
print("- `days_since_event`: Number of days elapsed since the event date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Filtering by Date Range
# MAGIC
# MAGIC **Goal:** Filter rows based on specific date criteria.
# MAGIC
# MAGIC **Example:** Filter events that occurred in the year 2023.

# COMMAND ----------

print("===== FILTERING BY DATE =====")

# Filter for events in 2023
events_2023_df = datetime_components_df.filter(col("year") == 2023)

display(events_2023_df.select("emp_id", "name", "event_timestamp", "year"))

print(f"\nTotal events in 2023: {events_2023_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Parsing Strings to Dates
# MAGIC
# MAGIC **Goal:** Convert string columns with custom formats into standard DateType.
# MAGIC
# MAGIC **Function:** `to_date(col, format)`

# COMMAND ----------

print("===== PARSING STRINGS TO DATES =====")

# Example: Parsing 'MM/dd/yyyy' string to DateType
string_date_data = [
    (1, "01/15/2023"),
    (2, "05/20/2022"),
    (3, "12/31/2023")
]
string_schema = "id INT, date_str STRING"
string_df = spark.createDataFrame(string_date_data, schema=string_schema)

parsed_df = string_df.withColumn("parsed_date", to_date(col("date_str"), "MM/dd/yyyy"))

display(parsed_df)

print("\nInterpretation:")
print("- `date_str`: Original string format")
print("- `parsed_date`: Standard PySpark DateType")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Summary - Date and Time Functions
# MAGIC
# MAGIC **Quick Reference:**
# MAGIC
# MAGIC | Function | Description | Example |
# MAGIC |---|---|---|
# MAGIC | `to_timestamp` | Convert string/numeric to timestamp | `to_timestamp(col)` |
# MAGIC | `date_format` | Format date to string | `date_format(col, "MM-dd-yyyy")` |
# MAGIC | `year`, `month` | Extract components | `year(col)` |
# MAGIC | `to_date` | Convert timestamp/string to date | `to_date(col)` |
# MAGIC | `date_add` | Add days to date | `date_add(col, 5)` |
# MAGIC | `datediff` | Difference in days | `datediff(end, start)` |
# MAGIC | `current_date` | Get today's date | `current_date()` |
# MAGIC | `add_months` | Add months to date | `add_months(col, 1)` |

# COMMAND ----------

print("===== DATE & TIME OPERATIONS SUMMARY =====")
print("✓ Created DataFrame with timestamps")
print("✓ Converted numeric timestamps to TimestampType")
print("✓ Formatted dates into custom strings")
print("✓ Extracted year, month, day components")
print("✓ Performed date arithmetic (add/subtract days/months)")
print("✓ Calculated date differences")
print("✓ Filtered data by year")
print("✓ Parsed custom string dates")

print("\n========================================")
print("✓ DATE & TIME MANIPULATION COMPLETE!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook:
# MAGIC
# MAGIC 1. **Copy the content** into a Databricks Notebook
# MAGIC 2. **Run All Cells** to see the transformation pipeline
# MAGIC 3. **Explore Outputs** using the `display()` tables
# MAGIC
# MAGIC ## Key PySpark Functions:
# MAGIC
# MAGIC - `pyspark.sql.functions.to_timestamp`
# MAGIC - `pyspark.sql.functions.date_format`
# MAGIC - `pyspark.sql.functions.date_add`
# MAGIC - `pyspark.sql.functions.datediff`
# MAGIC - `pyspark.sql.types.TimestampType`
