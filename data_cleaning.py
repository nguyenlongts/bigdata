from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import re

INPUT_CSV = "file:///app/data/raw/VN_housing_dataset.csv"

OUTPUT_PATH = "hdfs://namenode:9000/bigdata/cleaned_housing"



spark = (
    SparkSession.builder
    .appName("VietnamHousingCleaning")
    .master("local[*]")
    .config("spark.driver.memory", "2g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")



print("1.Loading raw dataset...")

df = spark.read.csv(
    INPUT_CSV,
    header=True,
    inferSchema=False,
    encoding="UTF-8"
)

print(f"Rows: {df.count()}")
print(f"Columns: {len(df.columns)}")

if "_c0" in df.columns:
    df = df.drop("_c0")
    print("Removed column: _c0")



print("\n2. Renaming columns...")

df = (
    df
    .withColumnRenamed("Ngày", "date")
    .withColumnRenamed("Địa chỉ", "address")
    .withColumnRenamed("Quận", "district")
    .withColumnRenamed("Huyện", "ward")
    .withColumnRenamed("Loại hình nhà ở", "house_type")
    .withColumnRenamed("Giấy tờ pháp lý", "legal_status")
    .withColumnRenamed("Số tầng", "floors")
    .withColumnRenamed("Số phòng ngủ", "bedrooms")
    .withColumnRenamed("Diện tích", "area")
    .withColumnRenamed("Dài", "length")
    .withColumnRenamed("Rộng", "width")
    .withColumnRenamed("Giá/m2", "price_per_m2")
)

print("Columns renamed successfully")


df = df.filter(
    F.col("price_per_m2").rlike("triệu|tỷ")
)


def extract_price(value):

    if value is None:
        return None

    v = str(value).lower().strip()

    match = re.findall(r"[\d,.]+", v)

    if not match:
        return None

    raw = match[0]

    try:
        if "triệu" in v:

            num = float(
                raw.replace(".", "")
                   .replace(",", ".")
            )

            return num
        elif "tỷ" in v:

            num = float(
                raw.replace(".", "")
                   .replace(",", ".")
            )

            return num * 1000

        else:
            return None

    except:
        return None


def extract_float(value):

    if value is None:
        return None

    v = str(value).lower().strip()

    match = re.findall(r"[\d,.]+", v)

    if not match:
        return None

    raw = match[0]

    try:

        num = float(
            raw.replace(".", "")
               .replace(",", ".")
        )

        return num

    except:
        return None


def extract_int(value):

    if value is None:
        return None

    v = str(value).lower()

    if "nhiều hơn" in v:
        return 10

    match = re.findall(r"\d+", v)

    if not match:
        return None

    return int(match[0])



price_udf = F.udf(extract_price, DoubleType())

float_udf = F.udf(extract_float, DoubleType())

int_udf = F.udf(extract_int, IntegerType())



print("\n3. Cleaning price column...")

df = df.withColumn(
    "price_per_m2",
    price_udf(F.col("price_per_m2"))
)
print("Cleaned price_per_m2 column")


FLOAT_COLS = [
    "area",
    "length",
    "width",
    "floors"
]

for col in FLOAT_COLS:

    if col in df.columns:

        df = df.withColumn(
            col,
            float_udf(F.col(col))
        )

        print(f"Cleaned float column: {col}")



INT_COLS = [
    "bedrooms"
]

for col in INT_COLS:

    if col in df.columns:

        df = df.withColumn(
            col,
            int_udf(F.col(col))
        )

        print(f"Cleaned integer column: {col}")


print("\n4. Removing invalid rows...")

before = df.count()

df = df.dropna(
    subset=[
        "price_per_m2",
        "area"
    ]
)

df = df.filter(F.col("area") > 0)

df = df.filter(F.col("price_per_m2") >= 1)

df = df.filter(F.col("price_per_m2") <= 5000)

after = df.count()

print(f"Removed {before - after} invalid rows")


print("\n5. Removing duplicate rows...")

before = df.count()

df = df.dropDuplicates()

after = df.count()

print(f"Removed {before - after} duplicate rows")

print("\n--- Schema ---")

df.printSchema()


print("\n--- Missing Values ---")

missing_df = df.select([
    F.count(
        F.when(F.col(c).isNull(), c)
    ).alias(c)
    for c in df.columns
])

missing_df.show(truncate=False)



print("\n--- Price Statistics (triệu/m²) ---")

df.select("price_per_m2").describe().show()

print(f"\n6. Saving dataset to HDFS...")

df.write.mode("overwrite").parquet(
    OUTPUT_PATH
)

print(f"Saved to: {OUTPUT_PATH}")



print("\n7. Reading back from HDFS...")

df_loaded = spark.read.parquet(
    OUTPUT_PATH
)

print(f"Loaded rows: {df_loaded.count()}")

print("\n--- Sample Rows ---")

df_loaded.show(10, truncate=False)

spark.stop()

print("\nDone")