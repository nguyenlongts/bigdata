import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")

os.makedirs("output", exist_ok=True)


eda = pd.read_csv("data/eda_data.csv")

preds = pd.read_csv("data/predictions.csv")

models = pd.read_csv("data/model_results.csv")

fi_df = pd.read_csv("data/feature_importances.csv")




FEATURE_LABEL = {
    "area": "Diện tích",
    "length": "Chiều dài",
    "width": "Chiều rộng",
    "floors": "Số tầng",
    "bedrooms": "Phòng ngủ",
    "total_rooms": "Tổng phòng",
    "has_legal": "Pháp lý",
    "district_idx": "Quận",
    "ward_idx": "Phường",
    "house_type_idx": "Loại nhà"
}




fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    eda["total_price_million"].dropna(),
    bins=50,
    color="steelblue"
)

axes[0].set_title("Phân phối Tổng Giá Nhà")

axes[0].set_xlabel("Triệu VNĐ")

axes[0].set_ylabel("Số lượng")


axes[1].hist(
    eda["price_per_m2"].dropna(),
    bins=50,
    color="coral"
)

axes[1].set_title("Phân phối Giá/m²")

axes[1].set_xlabel("Triệu VNĐ / m²")

axes[1].set_ylabel("Số lượng")


plt.tight_layout()

plt.savefig(
    "output/01_price_distribution.png",
    dpi=150
)

plt.close()

print("[OK] 01_price_distribution.png")




district_data = (
    eda
    .groupby("district")["total_price_million"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

fig, ax = plt.subplots(figsize=(10, 6))

district_data.plot(
    kind="bar",
    ax=ax,
    color="skyblue"
)

ax.set_title("Giá Trung Bình Theo Quận")

ax.set_ylabel("Triệu VNĐ")

ax.set_xlabel("Quận")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "output/02_avg_price_by_district.png",
    dpi=150
)

plt.close()

print("[OK] 02_avg_price_by_district.png")




fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=eda,
    x="legal_status",
    y="total_price_million",
    ax=ax
)

ax.set_title("Giá Nhà Theo Pháp Lý")

ax.set_xlabel("Pháp lý")

ax.set_ylabel("Triệu VNĐ")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "output/03_price_by_legal.png",
    dpi=150
)

plt.close()

print("[OK] 03_price_by_legal.png")




sample = eda.sample(
    n=min(3000, len(eda)),
    random_state=42
)

fig, ax = plt.subplots(figsize=(10, 6))

scatter = ax.scatter(
    sample["area"],
    sample["total_price_million"],
    c=sample["bedrooms"],
    cmap="viridis",
    alpha=0.6
)

plt.colorbar(
    scatter,
    ax=ax,
    label="Bedrooms"
)

ax.set_title("Diện tích vs Tổng Giá")

ax.set_xlabel("Diện tích")

ax.set_ylabel("Triệu VNĐ")

plt.tight_layout()

plt.savefig(
    "output/04_area_vs_price.png",
    dpi=150
)

plt.close()

print("[OK] 04_area_vs_price.png")




corr_cols = [
    "area",
    "length",
    "width",
    "floors",
    "bedrooms",
    "price_per_m2",
    "total_price_million"
]

corr = eda[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

ax.set_title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "output/05_correlation_heatmap.png",
    dpi=150
)

plt.close()

print("[OK] 05_correlation_heatmap.png")




fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ["rmse", "r2", "mae"]

titles = ["RMSE", "R²", "MAE"]

for ax, metric, title in zip(axes, metrics, titles):

    ax.bar(
        models["model"],
        models[metric]
    )

    ax.set_title(title)

    ax.tick_params(
        axis="x",
        rotation=15
    )

plt.tight_layout()

plt.savefig(
    "output/06_model_comparison.png",
    dpi=150
)

plt.close()

print("[OK] 06_model_comparison.png")




fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    preds["total_price_million"],
    preds["prediction"],
    alpha=0.5
)

min_val = min(
    preds["total_price_million"].min(),
    preds["prediction"].min()
)

max_val = max(
    preds["total_price_million"].max(),
    preds["prediction"].max()
)

ax.plot(
    [min_val, max_val],
    [min_val, max_val],
    "r--"
)

ax.set_title("Actual vs Predicted")

ax.set_xlabel("Actual")

ax.set_ylabel("Prediction")

plt.tight_layout()

plt.savefig(
    "output/07_actual_vs_prediction.png",
    dpi=150
)

plt.close()

print("[OK] 07_actual_vs_prediction.png")




fi_df["label"] = (
    fi_df["feature"]
    .map(FEATURE_LABEL)
    .fillna(fi_df["feature"])
)

fi_df = fi_df.sort_values("importance")

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    fi_df["label"],
    fi_df["importance"]
)

ax.set_title("Feature Importance")

ax.set_xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "output/08_feature_importance.png",
    dpi=150
)

plt.close()

print("[OK] 08_feature_importance.png")


print("\n[DONE] All charts saved.")