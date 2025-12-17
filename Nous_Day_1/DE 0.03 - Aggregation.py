# Databricks notebook source
# MAGIC %md
# MAGIC # Employee Aggregation Analysis - Databricks Notebook
# MAGIC
# MAGIC This notebook demonstrates comprehensive aggregation operations on an employee dataset using PySpark.
# MAGIC Each section includes clear explanations and practical examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 1: Import Required Libraries

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg, min, max, count, approx_count_distinct, cos, sqrt, expr, upper

print("✓ All required libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 2: Create Employee DataFrame (20 Rows)
# MAGIC
# MAGIC This cell creates a sample employee dataset with the following columns:
# MAGIC - emp_id: Employee ID
# MAGIC - name: Employee Name
# MAGIC - dept: Department (IT, HR, Finance, Sales)
# MAGIC - salary: Annual Salary
# MAGIC - country: Country Code (IN, US, UK, DE, CA)
# MAGIC - experience: Years of Experience
# MAGIC - is_active: Active Status (True/False/None)

# COMMAND ----------

# Initialize Spark Session
spark = SparkSession.builder.appName("EmployeeAggregation").getOrCreate()

# Sample Employee Data
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

# Define Schema
schema = "emp_id INT, name STRING, dept STRING, salary INT, country STRING, experience INT, is_active BOOLEAN"

# Create DataFrame
df = spark.createDataFrame(data, schema=schema)

# Display the DataFrame
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 3: Explore DataFrame Schema & Basic Info
# MAGIC
# MAGIC Let's understand the structure and basic statistics of our employee dataset.

# COMMAND ----------

# Display Schema
print("===== DATAFRAME SCHEMA =====")
df.printSchema()

# Show basic statistics
print("\n===== BASIC STATISTICS =====")
print(f"Total Rows: {df.count()}")
print(f"Total Columns: {len(df.columns)}")
print(f"Column Names: {df.columns}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 4: Basic GroupBy - Group by Department
# MAGIC
# MAGIC **What is GroupBy?**
# MAGIC GROUP BY is used to group rows that have the same values in specified columns.
# MAGIC This creates a GroupedData object that can be aggregated.

# COMMAND ----------

print("===== BASIC GROUPBY OPERATION =====")
print("Grouping employees by DEPARTMENT\n")

df_grouped_dept = df.groupBy("dept")
print(f"GroupedData object type: {type(df_grouped_dept)}")
print("This object is used for aggregation operations")

# Show count grouped by department
df_group_dept_count = df.groupBy("dept").count()
display(df_group_dept_count)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 5: GroupBy Multiple Columns - Country & Department
# MAGIC
# MAGIC **Why Group by Multiple Columns?**
# MAGIC This helps analyze data across multiple dimensions simultaneously.

# COMMAND ----------

print("===== GROUPBY MULTIPLE COLUMNS =====")
print("Grouping employees by COUNTRY and DEPARTMENT\n")

df_group_multi = df.groupBy("country", "dept").count()
display(df_group_multi)

print("\nThis shows the count of employees in each (country, department) combination")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 6: COUNT Aggregation - Employees per Department
# MAGIC
# MAGIC **COUNT Function:**
# MAGIC COUNT returns the number of rows in each group.
# MAGIC Useful for headcount analysis by department.

# COMMAND ----------

print("===== COUNT AGGREGATION =====")
print("Counting total employees per DEPARTMENT\n")

employee_count_by_dept = df.groupBy("dept").count().withColumnRenamed("count", "total_employees")
display(employee_count_by_dept)

print("\nInterpretation:")
print("- HR has X employees")
print("- IT has X employees")
print("- Finance has X employees")
print("- Sales has X employees")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 7: AVG Aggregation - Average Salary by Country
# MAGIC
# MAGIC **AVG Function:**
# MAGIC AVG calculates the average/mean value of a numeric column per group.
# MAGIC Useful for salary benchmarking across regions.

# COMMAND ----------

print("===== AVG AGGREGATION =====")
print("Calculating AVERAGE SALARY by COUNTRY\n")

avg_salary_by_country = df.groupBy("country").avg("salary").withColumnRenamed("avg(salary)", "average_salary")
display(avg_salary_by_country)

print("\nInterpretation:")
print("- Shows the mean salary for employees in each country")
print("- Helps identify salary differences across regions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 8: SUM Aggregation - Total Salary & Experience by Country & Department
# MAGIC
# MAGIC **SUM Function:**
# MAGIC SUM calculates the total of a numeric column per group.
# MAGIC Useful for budget planning and resource allocation.

# COMMAND ----------

print("===== SUM AGGREGATION =====")
print("Calculating TOTAL SALARY and TOTAL EXPERIENCE by COUNTRY & DEPARTMENT\n")

sum_by_country_dept = df.groupBy("country", "dept").sum("salary", "experience") \
    .withColumnRenamed("sum(salary)", "total_salary") \
    .withColumnRenamed("sum(experience)", "total_experience_years")
    
display(sum_by_country_dept)

print("\nInterpretation:")
print("- Shows aggregated salary and experience for each (country, department) pair")
print("- Useful for understanding team composition and resource allocation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 9: MIN & MAX Aggregations - Salary Range by Department
# MAGIC
# MAGIC **MIN & MAX Functions:**
# MAGIC MIN returns the minimum value, MAX returns the maximum value per group.
# MAGIC Useful for understanding salary distribution and disparities.

# COMMAND ----------

print("===== MIN & MAX AGGREGATION =====")
print("Finding MINIMUM and MAXIMUM SALARY by DEPARTMENT\n")

salary_range_by_dept = df.groupBy("dept").agg(
    min("salary").alias("min_salary"),
    max("salary").alias("max_salary"),
    avg("salary").alias("avg_salary")
)

display(salary_range_by_dept)

print("\nInterpretation:")
print("- Shows salary range (min to max) for each department")
print("- Helps identify salary disparities within departments")
print("- Average salary gives the middle point")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 10: Complex Aggregation with .agg() and Aliases
# MAGIC
# MAGIC **.agg() Method:**
# MAGIC Allows multiple aggregation functions in a single call for cleaner code.
# MAGIC
# MAGIC **.alias() Method:**
# MAGIC Renames output columns for better readability and clarity.

# COMMAND ----------

print("===== COMPLEX AGGREGATION WITH .agg() =====")
print("Multiple aggregations with custom column names\n")

complex_agg = df.groupBy("country").agg(
    count("emp_id").alias("employee_count"),
    sum("salary").alias("total_salary_pool"),
    avg("salary").alias("average_salary"),
    min("experience").alias("min_years_experience"),
    max("experience").alias("max_years_experience")
)

display(complex_agg)

print("\nInterpretation:")
print("- Provides comprehensive metrics for each country")
print("- Shows both headcount and compensation data")
print("- Experience range indicates team seniority")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 11: Advanced Aggregation - Distinct Count
# MAGIC
# MAGIC **APPROX_COUNT_DISTINCT Function:**
# MAGIC Returns approximate distinct count of a column.
# MAGIC Faster than exact COUNT(DISTINCT ...) for large datasets.

# COMMAND ----------

print("===== ADVANCED AGGREGATION: DISTINCT COUNTS =====")
print("Approximate distinct employees per country\n")

country_analysis = df.groupBy("country").agg(
    count("emp_id").alias("total_employees"),
    approx_count_distinct("dept").alias("distinct_departments"),
    avg("experience").alias("avg_experience_years"),
    count(col("is_active")).alias("active_count")
)

display(country_analysis)

print("\nInterpretation:")
print("- Shows how many unique departments operate in each country")
print("- Indicates organizational structure")
print("- Average experience shows team maturity")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 12: Aggregation by Department with Active Employee Filter
# MAGIC
# MAGIC **WHERE/FILTER Clause:**
# MAGIC Apply conditions BEFORE grouping for targeted analysis.
# MAGIC Filters rows first, then aggregates the filtered data.

# COMMAND ----------

print("===== FILTERED AGGREGATION =====")
print("Aggregation ONLY for ACTIVE employees by DEPARTMENT\n")

active_emp_by_dept = df.filter(col("is_active") == True).groupBy("dept").agg(
    count("emp_id").alias("active_employees"),
    sum("salary").alias("active_salary_budget"),
    avg("experience").alias("avg_experience"),
    min("salary").alias("min_salary"),
    max("salary").alias("max_salary")
)

display(active_emp_by_dept)

print("\nInterpretation:")
print("- Shows metrics only for active employees")
print("- Useful for workforce planning and budgeting")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 13: Aggregation with HAVING Clause (Filter Groups)
# MAGIC
# MAGIC **HAVING Clause:**
# MAGIC Filters groups based on aggregated values.
# MAGIC Similar to WHERE, but applied AFTER grouping and aggregation.

# COMMAND ----------

print("===== AGGREGATION WITH HAVING CLAUSE =====")
print("Show departments with AVERAGE SALARY > 75000\n")

high_salary_depts = df.groupBy("dept").agg(
    count("emp_id").alias("employee_count"),
    avg("salary").alias("avg_salary"),
    sum("salary").alias("total_salary")
).filter(col("avg_salary") > 75000)

display(high_salary_depts)

print("\nInterpretation:")
print("- Filters groups (departments) based on aggregated values")
print("- Shows only high-paying departments")
print("- Useful for identifying premium divisions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 14: Mathematical Functions with Data
# MAGIC
# MAGIC **Mathematical Functions:**
# MAGIC PySpark supports mathematical operations like:
# MAGIC - sqrt(): Square root
# MAGIC - cos(), sin(): Trigonometric functions
# MAGIC - exp(), log(): Exponential and logarithm
# MAGIC - abs(), round(): Rounding functions

# COMMAND ----------

print("===== MATHEMATICAL FUNCTIONS =====")
print("Applying mathematical functions to emp_id\n")

# Create a small range DataFrame to demonstrate
math_demo = spark.range(10).select("id") \
    .withColumn("sqrt_id", sqrt("id")) \
    .withColumn("cos_id", cos("id"))

display(math_demo)

print("\nInterpretation:")
print("- sqrt(): Square root of the value")
print("- cos(): Cosine function (in radians)")
print("- Similar functions: sin(), exp(), log(), etc.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 15: Creating Derived Columns for Analysis
# MAGIC
# MAGIC **Derived Columns:**
# MAGIC Create new calculated fields based on existing data.
# MAGIC Useful for creating salary bands, experience levels, and business rules.

# COMMAND ----------

print("===== DERIVED COLUMNS =====")
print("Creating new calculated columns for analysis\n")

df_with_derived = df.withColumn(
    "salary_band", 
    expr("""
        CASE 
            WHEN salary < 70000 THEN 'Entry Level'
            WHEN salary < 85000 THEN 'Mid Level'
            WHEN salary < 95000 THEN 'Senior Level'
            ELSE 'Executive Level'
        END
    """)
).withColumn(
    "experience_level",
    expr("""
        CASE 
            WHEN experience < 3 THEN 'Junior'
            WHEN experience < 6 THEN 'Mid'
            ELSE 'Senior'
        END
    """)
)

# Aggregate by derived columns
display(df_with_derived.select("emp_id", "name", "salary", "salary_band", "experience", "experience_level"))

print("\n--- Aggregation by Salary Band ---")
display(df_with_derived.groupBy("salary_band").agg(
    count("emp_id").alias("count"),
    avg("salary").alias("avg_salary")
))

print("\nInterpretation:")
print("- Salary Band: Categorizes employees into compensation levels")
print("- Experience Level: Categorizes by career progression")
print("- Useful for HR analytics and workforce segmentation")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 16: Summary Statistics for Entire DataFrame
# MAGIC
# MAGIC **Summary Statistics:**
# MAGIC Get comprehensive statistics for numeric columns.
# MAGIC Useful for quick KPI generation and dashboard summaries.

# COMMAND ----------

print("===== SUMMARY STATISTICS =====")
print("Overall statistics for the entire employee dataset\n")

display(df.describe("salary", "experience"))

print("\nAlternative: Using .select() with multiple agg() functions")
stats_summary = df.select(
    count("emp_id").alias("total_employees"),
    count(col("is_active")).alias("active_records"),
    avg("salary").alias("avg_salary"),
    min("salary").alias("min_salary"),
    max("salary").alias("max_salary"),
    avg("experience").alias("avg_experience"),
    min("experience").alias("min_experience"),
    max("experience").alias("max_experience")
)

display(stats_summary)

print("\nInterpretation:")
print("- Provides a snapshot of the entire employee dataset")
print("- Useful for KPI dashboards and executive summaries")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cell 17: Final Summary - Aggregation Concepts
# MAGIC
# MAGIC **Quick Reference Guide for All Aggregation Operations**

# COMMAND ----------

print("===== AGGREGATION CONCEPTS SUMMARY =====\n")

print("1. GROUP BY: Groups rows by specified columns")
print("   Example: df.groupBy('dept').count()\n")

print("2. COUNT: Counts the number of rows in each group")
print("   Example: df.groupBy('dept').count()\n")

print("3. SUM: Sums numeric values in each group")
print("   Example: df.groupBy('dept').sum('salary')\n")

print("4. AVG: Calculates average of numeric values in each group")
print("   Example: df.groupBy('dept').avg('salary')\n")

print("5. MIN/MAX: Finds minimum/maximum values in each group")
print("   Example: df.groupBy('dept').agg(min('salary'), max('salary'))\n")

print("6. .agg(): Allows multiple aggregations in one call")
print("   Example: df.groupBy('dept').agg(sum('salary'), avg('salary'))\n")

print("7. .alias(): Renames output columns for clarity")
print("   Example: sum('salary').alias('total_salary')\n")

print("8. FILTER: Applied before grouping to filter rows")
print("   Example: df.filter(col('is_active') == True).groupBy('dept').count()\n")

print("9. HAVING: Filters groups based on aggregated values")
print("   Example: .filter(col('avg_salary') > 75000)\n")

print("10. Derived Columns: Create new columns from existing data for analysis")
print("    Example: .withColumn('salary_band', expr(CASE ... END))\n")

print("========================================")
print("✓ END OF NOTEBOOK - All Aggregation Concepts Covered!")
print("========================================")

# COMMAND ----------

# MAGIC %md
# MAGIC ## How to Use This Notebook in Databricks:
# MAGIC
# MAGIC 1. **Copy the entire content** into a Databricks Notebook
# MAGIC 2. **Each `# MAGIC %md` section** will render as formatted markdown documentation
# MAGIC 3. **Each code section** will execute as Python code cells
# MAGIC 4. **The `# COMMAND ----------` lines** automatically separate cells
# MAGIC 5. **Run All** to execute the entire notebook sequentially
# MAGIC 6. **Or run individual cells** using Shift + Enter
# MAGIC 7. **Rich `display()` outputs** will show formatted tables and visualizations
# MAGIC
# MAGIC ## Magic Commands Used:
# MAGIC
# MAGIC - `# MAGIC %md` - Markdown cells for documentation
# MAGIC - `# COMMAND ----------` - Cell separator
# MAGIC - `display()` - Rich visualization in Databricks
# MAGIC - `print()` - Console output
