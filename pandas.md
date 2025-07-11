# Essential Pandas Functions for ML Beginners

If you're starting with pandas for machine learning, mastering these essential functions will help you effectively load, explore, clean, and preprocess your data before feeding it into ML models.

---

## 1. Data Loading and Saving
- `pd.read_csv()` / `pd.read_excel()` — Load datasets from CSV or Excel files.
- `df.to_csv()` / `df.to_excel()` — Save processed DataFrames to files.

## 2. Exploring Data
- `df.head()` / `df.tail()` — View the first or last few rows.
- `df.info()` — Summary of DataFrame including data types and null counts.
- `df.describe()` — Statistical summary for numeric columns.
- `df.shape` — Get number of rows and columns.
- `df.columns` — List column names.
- `df.dtypes` — Check data types of each column.

## 3. Selecting and Filtering
- `df['col']` or `df.col` — Select a single column.
- `df.loc[]` / `df.iloc[]` — Select rows and columns by label or position.
- Boolean indexing — Filter rows by condition, e.g. `df[df['col'] > 0]`.

## 4. Data Cleaning
- `df.isnull()` / `df.isnull().sum()` — Detect missing values.
- `df.dropna()` — Drop rows with missing values.
- `df.fillna()` — Fill missing values with a specified value or method.
- `df.duplicated()` / `df.drop_duplicates()` — Detect and remove duplicate rows.
- `df.astype()` — Convert column data types.

## 5. Data Transformation
- `df.apply()` — Apply a function along rows or columns.
- `df.map()` / `df.replace()` — Map or replace values in a Series.
- `df.rename()` — Rename columns or indices.
- `df.sort_values()` — Sort data by column(s).
- `df.groupby()` — Group data for aggregation.
- `pd.get_dummies()` — One-hot encode categorical variables.
- `df.drop()` — Remove rows or columns.

## 6. Merging and Joining
- `pd.concat()` — Concatenate multiple DataFrames vertically or horizontally.
- `pd.merge()` — Join DataFrames on keys.

## 7. Statistical Operations
- `df.mean()`, `df.median()`, `df.mode()`, `df.std()` — Basic statistics.
- `df.corr()` — Compute correlation between columns.

---

## Why These Functions?

- Cover all core steps in data handling: loading, inspecting, cleaning, transforming, and preparing.
- Critical for building clean datasets ready for ML modeling.
- Strong data preparation skills can greatly improve ML success.


