import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os

# === Indlæs data ===
student_data = pd.read_csv("student_lifestyle_dataset.csv")

print(student_data.info())
print(student_data.head())

# === Mål og features ===
TARGET = "GPA"                  # Vi forudsiger GPA (kontinuerlig)
DROP_COLS = ["Student_ID"]      # ID skal ikke bruges som feature

X = student_data.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
y = student_data[TARGET]

# === Kolonnelister ===
num_feats = X.select_dtypes(include=["number"]).columns.tolist()
cat_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

# === Preprocessing ===
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_feats),
    ("cat", cat_pipe, cat_feats)
])

# === Model ===
model = Pipeline([
    ("preprocess", preprocess),
    ("reg", LinearRegression())
])

# === Split + CV ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")
print("\nCross-validation (Linear Regression on GPA):")
print("Scores:", scores)
print("Mean R^2:", scores.mean())
print("Std:", scores.std())

# Beregn korrelation (kun numeriske kolonner)
corr_matrix = student_data.select_dtypes(include=[np.number]).corr()

# Udskriv korrelation med GPA
print(corr_matrix["GPA"].sort_values(ascending=False))

plt.figure(figsize=(10, 8))
plt.matshow(corr_matrix, cmap="coolwarm", fignum=1)
plt.colorbar()
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title("Correlation Heatmap (Student Lifestyle Data)", pad=20)
# Ensure plots directory exists and save the heatmap and overall histograms instead of blocking with show()
SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# Save correlation heatmap
fig = plt.gcf()
fig.savefig(os.path.join(SAVE_DIR, "correlation_heatmap.png"))
plt.close(fig)

# Save overall histograms
student_data.hist(figsize=(12, 10))
fig = plt.gcf()
fig.savefig(os.path.join(SAVE_DIR, "all_histograms.png"))
plt.close(fig)


def save_feature_distributions(df, target_col="GPA", drop_cols=None, out_dir="plots"):
    if drop_cols is None:
        drop_cols = []

    os.makedirs(out_dir, exist_ok=True)

    cols = [c for c in df.columns if c not in drop_cols]
    for col in cols:
        try:
            plt.clf()
            if pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(6, 4))
                plt.hist(df[col].dropna(), bins=30, color="tab:blue", edgecolor="black")
                plt.title(f"Histogram of {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                fname = os.path.join(out_dir, f"hist_{col}.png")
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
            else:
                # categorical: show value counts as bar plot
                counts = df[col].value_counts(dropna=False)
                plt.figure(figsize=(6, 4))
                counts.plot(kind="bar", color="tab:orange", edgecolor="black")
                plt.title(f"Value counts for {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                fname = os.path.join(out_dir, f"bar_{col}.png")
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
        except Exception as e:
            print(f"Failed to plot {col}: {e}")


# Save plots for all features except Student_ID (if present)
SAVE_DIR = "plots"
DROP_PLOT_COLS = ["Student_ID"]
save_feature_distributions(student_data, target_col=TARGET, drop_cols=DROP_PLOT_COLS, out_dir=SAVE_DIR)


def save_scatter_plots(df, target_col="GPA", drop_cols=None, out_dir="plots/scatter"):
    """Create plots relating features to target and save PNGs.

    Behavior depends on whether target_col is numeric or categorical:
    - If numeric: create scatter plots (feature vs target) for numeric features and boxplots of target by categorical features.
    - If categorical: create boxplots of numeric features grouped by target and stacked count bars for categorical features.
    """
    if drop_cols is None:
        drop_cols = []

    os.makedirs(out_dir, exist_ok=True)

    cols = [c for c in df.columns if c not in drop_cols and c != target_col]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    target_is_numeric = pd.api.types.is_numeric_dtype(df[target_col])

    if target_is_numeric:
        # Numeric target: scatter plots for numeric features vs target
        for col in numeric_cols:
            try:
                ax = df.plot.scatter(x=col, y=target_col, figsize=(6, 4), alpha=0.6)
                ax.set_title(f"{target_col} vs {col}")
                fname = os.path.join(out_dir, f"scatter_{col}_vs_{target_col}.png")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(fname)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create scatter for {col}: {e}")

        # Categorical features: boxplot of target by category
        for col in categorical_cols:
            try:
                ax = df.boxplot(column=target_col, by=col, figsize=(8, 5))
                plt.suptitle("")
                ax.set_title(f"{target_col} by {col}")
                plt.xlabel(col)
                plt.ylabel(target_col)
                fname = os.path.join(out_dir, f"box_{target_col}_by_{col}.png")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(fname)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create boxplot for {col}: {e}")

    else:
        # Categorical target: create boxplots of each numeric feature grouped by target
        for col in numeric_cols:
            try:
                ax = df.boxplot(column=col, by=target_col, figsize=(8, 5))
                plt.suptitle("")
                ax.set_title(f"{col} by {target_col}")
                plt.xlabel(target_col)
                plt.ylabel(col)
                fname = os.path.join(out_dir, f"box_{col}_by_{target_col}.png")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(fname)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create grouped boxplot for {col}: {e}")

        # For categorical features, show stacked counts of target within each category
        for col in categorical_cols:
            try:
                ct = pd.crosstab(df[col], df[target_col])
                ax = ct.plot(kind="bar", stacked=True, figsize=(8, 5))
                ax.set_title(f"Counts of {target_col} by {col}")
                ax.set_xlabel(col)
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                fname = os.path.join(out_dir, f"stacked_counts_{col}_by_{target_col}.png")
                fig = ax.get_figure()
                fig.tight_layout()
                fig.savefig(fname)
                plt.close(fig)
            except Exception as e:
                print(f"Failed to create stacked bar for {col}: {e}")


# Save plots using Stress_Level as the target instead of GPA
# Additionally: create scatter plots where Stress_Level is mapped to numeric codes
def save_scatter_with_categorical_target_as_numeric(df, cat_target_col, drop_cols=None, out_dir="plots/scatter"):
    """Map a categorical target to numeric codes and save scatter plots of numeric features vs the mapped codes.

    Adds small vertical jitter so points are visible (since mapped codes are integers).
    Filenames follow the pattern scatter_<feature>_vs_<cat_target_col>.png
    """
    if drop_cols is None:
        drop_cols = []

    os.makedirs(out_dir, exist_ok=True)

    # map categories to integer codes
    cat_series = df[cat_target_col].astype('category')
    codes = cat_series.cat.codes

    # mapping dict for legend/labels
    mapping = dict(enumerate(cat_series.cat.categories))

    cols = [c for c in df.columns if c not in drop_cols and c != cat_target_col]
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        try:
            x = df[col].values
            y = codes.values.astype(float)
            # add small jitter to y for visibility
            jitter = (np.random.rand(len(y)) - 0.5) * 0.1
            y_jitter = y + jitter

            plt.figure(figsize=(6, 4))
            plt.scatter(x, y_jitter, alpha=0.5, s=20)
            plt.yticks(list(mapping.keys()), list(mapping.values()))
            plt.xlabel(col)
            plt.ylabel(cat_target_col)
            plt.title(f"{cat_target_col} vs {col}")
            fname = os.path.join(out_dir, f"scatter_{col}_vs_{cat_target_col}.png")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
        except Exception as e:
            print(f"Failed to create categorical-mapped scatter for {col}: {e}")


# Call the original saver for Stress_Level (boxplots/stacked counts)
save_scatter_plots(student_data, target_col="Stress_Level", drop_cols=DROP_PLOT_COLS, out_dir=os.path.join(SAVE_DIR, "scatter"))

# And also save scatter plots with Stress_Level mapped to numeric codes
save_scatter_with_categorical_target_as_numeric(student_data, cat_target_col="Stress_Level", drop_cols=DROP_PLOT_COLS, out_dir=os.path.join(SAVE_DIR, "scatter"))


def save_two_feature_scatter_colored_by_category(df, x_col, y_col, category_col, out_dir="plots/scatter"):
    """Create a scatter plot of x_col vs y_col where each point is colored by category_col.

    Color mapping: Low -> green, Moderate -> yellow, High -> red. Saves PNG to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    # define color mapping (fallback to gray for unknowns)
    color_map = {
        "Low": "green",
        "Moderate": "yellow",
        "High": "red"
    }

    colors = df[category_col].map(color_map).fillna("gray")

    plt.figure(figsize=(7, 5))
    plt.scatter(df[x_col], df[y_col], c=colors, alpha=0.7, edgecolor='k', s=30)
    # create custom legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
    plt.legend(handles=patches, title=category_col)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col} colored by {category_col}")
    fname = os.path.join(out_dir, f"scatter_{x_col}_vs_{y_col}_colored_by_{category_col}.png")
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


# Save a Sleep vs GPA scatter colored by Stress_Level
save_two_feature_scatter_colored_by_category(student_data, x_col="Sleep_Hours_Per_Day", y_col="GPA", category_col="Stress_Level", out_dir=os.path.join(SAVE_DIR, "scatter"))


def save_pairwise_scatter_colored(df, numeric_cols, category_col="Stress_Level", out_dir="plots/scatter"):
    """Create scatter plots for every pair of numeric_cols colored by category_col and save PNGs.

    Filenames: scatter_<x>_vs_<y>_colored_by_<category_col>.png
    """
    os.makedirs(out_dir, exist_ok=True)

    color_map = {"Low": "green", "Moderate": "yellow", "High": "red"}

    for i, x in enumerate(numeric_cols):
        for j, y in enumerate(numeric_cols):
            if i == j:
                continue
            try:
                colors = df[category_col].map(color_map).fillna("gray")
                plt.figure(figsize=(6, 4))
                plt.scatter(df[x], df[y], c=colors, alpha=0.6, s=20, edgecolor='k')
                # legend
                import matplotlib.patches as mpatches
                patches = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
                plt.legend(handles=patches, title=category_col)
                plt.xlabel(x)
                plt.ylabel(y)
                plt.title(f"{y} vs {x} colored by {category_col}")
                fname = os.path.join(out_dir, f"scatter_{x}_vs_{y}_colored_by_{category_col}.png")
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
            except Exception as e:
                print(f"Failed pairwise scatter {x} vs {y}: {e}")


# Determine numeric columns (excluding Student_ID if present)
numeric_columns = student_data.drop(columns=DROP_PLOT_COLS, errors='ignore').select_dtypes(include=[np.number]).columns.tolist()
save_pairwise_scatter_colored(student_data, numeric_columns, category_col="Stress_Level", out_dir=os.path.join(SAVE_DIR, "scatter"))


def save_physical_activity_boxplot_and_outliers(df, col="Physical_Activity_Hours_Per_Day", out_dir="plots"):
    """Save a boxplot for the given column and detect outliers using the IQR method.

    Outliers are saved to CSV at plots/physical_activity_outliers.csv and a PNG boxplot is saved to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)

    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df[(df[col] < lower) | (df[col] > upper)][[col, "Student_ID"] if "Student_ID" in df.columns else [col]]

    # Save outliers to CSV
    out_csv = os.path.join(out_dir, "physical_activity_outliers.csv")
    outliers.to_csv(out_csv, index=False)

    # Save boxplot
    plt.figure(figsize=(6, 4))
    plt.boxplot(series, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
    plt.title(f"Boxplot of {col}")
    plt.ylabel(col)
    out_png = os.path.join(out_dir, f"boxplot_{col}.png")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"Saved boxplot to {out_png}")
    print(f"Detected {len(outliers)} outliers; saved to {out_csv}")


# Generate boxplot and outliers CSV for physical activity
save_physical_activity_boxplot_and_outliers(student_data, col="Physical_Activity_Hours_Per_Day", out_dir=SAVE_DIR)