"""
Task 2: EDA and Data Cleaning
Loads data/raw_data.csv, explores it, produces required plots,
cleans it with documented reasoning, and saves data/clean_data.csv.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

df = pd.read_csv("data/raw_data.csv")

# ------------------------------------------------------------------
# STEP 1: First look (see NOTES.md for the written summary)
# ------------------------------------------------------------------
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Exact duplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------------
# STEP 2: Visualize BEFORE cleaning
# ------------------------------------------------------------------

# 2a. Class balance plot
plt.figure(figsize=(10, 5))
order = df["primary_type"].value_counts().index
sns.countplot(data=df, x="primary_type", order=order, hue="primary_type",
              palette="viridis", legend=False)
plt.title("Class Balance: Number of Pokemon per Primary Type")
plt.xlabel("Primary Type")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("notebooks/01_class_balance.png", dpi=120)
plt.close()

# 2b. Correlation heatmap of numeric features
numeric_cols = ["height", "weight", "base_experience", "hp", "attack",
                 "defense", "special_attack", "special_defense", "speed"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True)
plt.title("Correlation Heatmap: Numeric Features")
plt.tight_layout()
plt.savefig("notebooks/02_correlation_heatmap.png", dpi=120)
plt.close()

# 2c. Relationship plot 1: speed by primary_type (boxplot)
plt.figure(figsize=(12, 5))
order_speed = df.groupby("primary_type")["speed"].median().sort_values(ascending=False).index
sns.boxplot(data=df, x="primary_type", y="speed", order=order_speed,
            hue="primary_type", palette="viridis", legend=False)
plt.title("Speed Distribution by Primary Type")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("notebooks/03_speed_by_type.png", dpi=120)
plt.close()

# 2d. Relationship plot 2: attack vs defense scatter, colored by type
plt.figure(figsize=(8, 6))
top6_types = df["primary_type"].value_counts().head(6).index
subset = df[df["primary_type"].isin(top6_types)]
sns.scatterplot(data=subset, x="attack", y="defense", hue="primary_type",
                 palette="tab10", alpha=0.7)
plt.title("Attack vs Defense (Top 6 Most Common Types)")
plt.tight_layout()
plt.savefig("notebooks/04_attack_vs_defense.png", dpi=120)
plt.close()

# 2e. Relationship plot 3: hp distribution overall (histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df["hp"], bins=30, kde=True, color="steelblue")
plt.title("Distribution of HP Stat (All Pokemon)")
plt.tight_layout()
plt.savefig("notebooks/05_hp_distribution.png", dpi=120)
plt.close()

print("Saved 5 plots to notebooks/")

# ------------------------------------------------------------------
# STEP 3: Clean, with justification (see NOTES.md for the writeup)
# ------------------------------------------------------------------

df_clean = df.copy()

# Decision 1: drop alternate forms (mega/gmax/regional/forme variants).
# These share an id > 10000 in PokeAPI's numbering and represent the
# same species as an existing base-form row already in the dataset.
# Keeping them would let one species "vote" multiple times and would
# also keep the fake gmax weight=10000 placeholder values in the data.
df_clean = df_clean[df_clean["id"] <= 10000].copy()

# Decision 2: base_experience has ~3.6% missing -> impute with median
# rather than drop rows, since we don't want to lose otherwise-good
# rows over one field. Median is used (not mean) because base_experience
# is right-skewed (a few very high values pull the mean up).
df_clean["base_experience"] = df_clean["base_experience"].fillna(
    df_clean["base_experience"].median()
)

# Decision 3: secondary_type is missing for single-typed Pokemon --
# this is a real category ("no second type"), not something to impute
# with a statistic. Fill with an explicit label instead.
df_clean["secondary_type"] = df_clean["secondary_type"].fillna("none")

# Decision 4: exact duplicate rows -- none were found, but drop defensively
# in case re-running the fetch script ever introduces any.
before = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Dropped {before - len(df_clean)} exact duplicate rows")

# Decision 5: rare classes -- check whether any primary_type has too few
# examples to survive a stratified train/test split later.
type_counts = df_clean["primary_type"].value_counts()
print("\nSmallest classes after form-filtering:")
print(type_counts.tail(5))
# (all classes have >= ~10 examples after filtering alt-forms out,
# so no merging into "other" is needed here -- see NOTES.md)

# Decision 6: categorical encoding for secondary_type (a FEATURE).
# primary_type is the TARGET and gets its own LabelEncoder in Task 3,
# so it is intentionally left as text here.
df_clean = pd.get_dummies(df_clean, columns=["secondary_type"],
                           prefix="sec_type")

print("\nFinal shape after cleaning:", df_clean.shape)

# ------------------------------------------------------------------
# STEP 4: Save
# ------------------------------------------------------------------
df_clean.to_csv("data/clean_data.csv", index=False)
print("Saved data/clean_data.csv")
