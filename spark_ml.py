from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    VectorAssembler,
    StringIndexer
)

from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor
)

from pyspark.ml.evaluation import RegressionEvaluator

import pandas as pd




INPUT_PATH = "hdfs://namenode:9000/bigdata/processed_housing"

OUTPUT_DIR = "/app/data"

TARGET = "total_price_million"

CATEGORICAL_COLS = [
    "district",
    "ward",
    "house_type"
]

FEATURE_COLS = [
    "area",
    "length",
    "width",
    "floors",
    "bedrooms",
    "total_rooms",
    "has_legal"
]


spark = (
    SparkSession.builder
    .appName("VietnamHousingML")
    .master("local[*]")
    .config("spark.driver.memory", "4g")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")



print("\n[1] Loading processed housing data...")

df = spark.read.parquet(INPUT_PATH)

print(f"Rows: {df.count()}")
print(f"Columns: {len(df.columns)}")

print("\n--- Schema ---")

df.printSchema()




df = df.dropna(subset=[TARGET])



print("\n[2] Encoding categorical columns...")

indexers = [
    StringIndexer(
        inputCol=col,
        outputCol=col + "_idx",
        handleInvalid="keep"
    )

    for col in CATEGORICAL_COLS
]

encoded_cols = [
    col + "_idx"
    for col in CATEGORICAL_COLS
]




ALL_FEATURES = FEATURE_COLS + encoded_cols

assembler = VectorAssembler(
    inputCols=ALL_FEATURES,
    outputCol="features",
    handleInvalid="skip"
)



print("\n[3] Splitting dataset 80/20...")

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print(f"Train rows: {train_df.count()}")
print(f"Test rows : {test_df.count()}")



ev_rmse = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="rmse"
)

ev_r2 = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="r2"
)

ev_mae = RegressionEvaluator(
    labelCol=TARGET,
    predictionCol="prediction",
    metricName="mae"
)

results = []




print("\n[4] Training Linear Regression...")

lr = LinearRegression(
    labelCol=TARGET,
    featuresCol="features",
    maxIter=100,
    regParam=0.1
)

pipe_lr = Pipeline(
    stages=indexers + [assembler, lr]
)

model_lr = pipe_lr.fit(train_df)

pred_lr = model_lr.transform(test_df)

lr_rmse = ev_rmse.evaluate(pred_lr)
lr_r2   = ev_r2.evaluate(pred_lr)
lr_mae  = ev_mae.evaluate(pred_lr)

print(f"RMSE: {lr_rmse:.4f}")
print(f"R2  : {lr_r2:.4f}")
print(f"MAE : {lr_mae:.4f}")

results.append((
    "Linear Regression",
    lr_rmse,
    lr_r2,
    lr_mae
))




print("\n[5] Training Decision Tree...")

dt = DecisionTreeRegressor(
    labelCol=TARGET,
    featuresCol="features",
    maxDepth=8,
    maxBins=512,
    seed=42
)

pipe_dt = Pipeline(
    stages=indexers + [assembler, dt]
)

model_dt = pipe_dt.fit(train_df)

pred_dt = model_dt.transform(test_df)

dt_rmse = ev_rmse.evaluate(pred_dt)
dt_r2   = ev_r2.evaluate(pred_dt)
dt_mae  = ev_mae.evaluate(pred_dt)

print(f"RMSE: {dt_rmse:.4f}")
print(f"R2  : {dt_r2:.4f}")
print(f"MAE : {dt_mae:.4f}")

results.append((
    "Decision Tree",
    dt_rmse,
    dt_r2,
    dt_mae
))




print("\n[6] Training Random Forest...")

rf = RandomForestRegressor(
    labelCol=TARGET,
    featuresCol="features",
    numTrees=100,
    maxBins=512,
    maxDepth=10,
    seed=42
)

pipe_rf = Pipeline(
    stages=indexers + [assembler, rf]
)

model_rf = pipe_rf.fit(train_df)

pred_rf = model_rf.transform(test_df)

rf_rmse = ev_rmse.evaluate(pred_rf)
rf_r2   = ev_r2.evaluate(pred_rf)
rf_mae  = ev_mae.evaluate(pred_rf)

print(f"RMSE: {rf_rmse:.4f}")
print(f"R2  : {rf_r2:.4f}")
print(f"MAE : {rf_mae:.4f}")

results.append((
    "Random Forest",
    rf_rmse,
    rf_r2,
    rf_mae
))



print("\n[7] Random Forest Feature Importance")

importances = (
    model_rf
    .stages[-1]
    .featureImportances
    .toArray()
)

fi_df = pd.DataFrame({
    "feature": ALL_FEATURES,
    "importance": importances
})

fi_df = fi_df.sort_values(
    "importance",
    ascending=False
)

print(fi_df.to_string(index=False))

fi_df.to_csv(
    f"{OUTPUT_DIR}/feature_importances.csv",
    index=False
)

print("\nSaved feature_importances.csv")




print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    f"{'Model':<22}"
    f"{'RMSE':>12}"
    f"{'R2':>12}"
    f"{'MAE':>12}"
)

print("-" * 60)

for model_name, rmse, r2, mae in results:

    print(
        f"{model_name:<22}"
        f"{rmse:>12.4f}"
        f"{r2:>12.4f}"
        f"{mae:>12.4f}"
    )

print("=" * 60)




res_df = pd.DataFrame(
    results,
    columns=["model", "rmse", "r2", "mae"]
)

res_df.to_csv(
    f"{OUTPUT_DIR}/model_results.csv",
    index=False
)

print("\nSaved model_results.csv")




best_model_name = res_df.loc[
    res_df["r2"].idxmax(),
    "model"
]

print(f"\n[8] Best Model: {best_model_name}")

prediction_map = {
    "Linear Regression": pred_lr,
    "Decision Tree": pred_dt,
    "Random Forest": pred_rf
}

best_predictions = prediction_map[best_model_name]


(
    best_predictions
    .select(
        "district",
        "ward",
        "area",
        "bedrooms",
        "floors",
        "price_per_m2",
        "total_price_million",
        "prediction"
    )
    .limit(1000)
    .toPandas()
    .to_csv(
        f"{OUTPUT_DIR}/predictions.csv",
        index=False
    )
)

print("Saved predictions.csv")



(
    df.select(
        "district",
        "ward",
        "house_type",
        "legal_status",
        "area",
        "length",
        "width",
        "floors",
        "bedrooms",
        "price_per_m2",
        "total_price_million"
    )
    .toPandas()
    .to_csv(
        f"{OUTPUT_DIR}/eda_data.csv",
        index=False
    )
)

print("Saved eda_data.csv")




print("\n--- Sample Predictions ---")

best_predictions.select(
    "district",
    "area",
    "bedrooms",
    "total_price_million",
    "prediction"
).show(10, truncate=False)




spark.stop()

print("\n[DONE] Spark ML pipeline complete.")

