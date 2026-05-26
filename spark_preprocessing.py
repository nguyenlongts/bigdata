from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType


INPUT_PATH = "hdfs://namenode:9000/bigdata/cleaned_housing"

OUTPUT_PATH = "hdfs://namenode:9000/bigdata/processed_housing"

NUMERIC_COLS = [
    "area",
    "length",
    "width",
    "floors",
    "bedrooms",
    "price_per_m2"
]

CATEGORICAL_COLS = [
    "district",
    "ward",
    "house_type",
    "legal_status"
]

spark = (
    SparkSession.builder
    .appName("VietnamHousingPreprocessing")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

print("1. Loading cleaned dataset from HDFS...")

df = spark.read.parquet(INPUT_PATH)

df.cache()

print(f"Rows: {df.count()}")
print(f"Columns: {len(df.columns)}")

print("\n--- Schema ---")

df.printSchema()

print("\n2. Filling missing values.")
print("\n2.1 Filling missing numeric values with median.")

for col in NUMERIC_COLS:

    quantiles = df.approxQuantile(col, [0.5], 0.05)

    if quantiles and quantiles[0] is not None:

        median = quantiles[0]

        df = df.fillna({col: median})

        print(f"{col}: median = {median}")

print("\n2.2 Filling missing categorical values = Unknown")

for col in CATEGORICAL_COLS:

    df = df.fillna({col: "Unknown"})


print("\n3. Removing price outliers using IQR.")

q1, q3 = df.approxQuantile(
    "price_per_m2",
    [0.25, 0.75],
    0.01
)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

before = df.count()

df = df.filter(
    (F.col("price_per_m2") >= lower) &
    (F.col("price_per_m2") <= upper)
)

after = df.count()

print(f"Price range kept: {lower:.2f} - {upper:.2f}")

print(f"Removed {before - after} outlier rows")


print("\n4. Creating new features.")
df = df.withColumn(
    "total_price_million",
    (
        F.col("price_per_m2") * F.col("area")
    ).cast(DoubleType())
)


df = df.withColumn(
    "total_rooms",
    (
        F.col("bedrooms") + F.col("floors")
    ).cast(IntegerType())
)


df = df.withColumn(
    "has_legal",
    F.when(
        F.col("legal_status") == "Đã có sổ",
        1
    ).otherwise(0)
)

print("Created features:")
print("- total_price_million")
print("- total_rooms")
print("- has_legal")


print(f"\n5. Saving parquet to: {OUTPUT_PATH}")

(
    df.write
    .mode("overwrite")
    .partitionBy("district")
    .parquet(OUTPUT_PATH)
)

print("Parquet saved successfully")

print("\n6. Reading parquet back to confirm.")

df_loaded = spark.read.parquet(OUTPUT_PATH)

print(f"Loaded rows: {df_loaded.count()}")

print("\n--- Sample Rows ---")
df_loaded.select(
    "district",
    "area",
    "bedrooms",
    "floors",
    "price_per_m2",
    "total_price_million",
    "total_rooms",
    "has_legal"
).show(10, truncate=False)

print("\n--- Average Price By District ---")

(
    df_loaded
    .groupBy("district")
    .agg(
        F.count("*").alias("count"),
        F.round(
            F.avg("price_per_m2"),
            2
        ).alias("avg_price_per_m2")
    )
    .orderBy(F.desc("count"))
    .show(10, truncate=False)
)

spark.stop()

print("\nDone.")