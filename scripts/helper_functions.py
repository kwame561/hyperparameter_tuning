#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def loguniform(low, high, size=None):
    import numpy as np
    return np.exp(np.random.uniform(low, high, size))


def print_df_shape(df):
    return print(f"data : {df.shape[0]:,} rows x {df.shape[1]:,} columns")


def describe_numeric_data(df):
    """Takes a PANDAS data frame and describes
    it with basic statistics, as long as all the 
    columns are numeric.
    """
    df_copy = df.copy()
    df_res = df_copy.describe().T
    df_res['dtypes'] = df_copy.dtypes
    df_res['NULLs'] = df_copy.isna().sum()
    print_df_shape(df_copy)
    return df_res


def records_by_column_value(df, col_name: str):
    import pandas as pd
    df_copy = df.copy()
    summary = df_copy[col_name].value_counts()
    summary_dict = {col_name: summary.index, 'records': summary.values}
    df_summary = pd.DataFrame.from_dict(summary_dict)
    df_summary['percent_of_toal'] = round(
        df_summary['records'] / df_copy.shape[0], 2)
    df_summary['cumsum_percent'] = df_summary['percent_of_toal'].cumsum()
    return print(df_summary)


def get_vif(df):
    """Calculates the variance inflation factor for each numeric variable."""
    import pandas as pd
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                       for i in range(len(df.columns))]
    over_5 = vif_data.query("VIF > 5")
    print("Features that appear to be highly correlated:")
    for i in over_5['feature']:
        print(f"\t{i}")
    print(f"\n{vif_data}")


def detecting_multicollinearity(df):
    """Calculates the severity of multicollinearity using
    the variance inflation factor technique."""
    df_copy = df.copy()
    df_copy = df_copy.select_dtypes(include='number')
    if df_copy.empty | len(df_copy.columns) < 2:
        print("Dataframe must have at least two numeric columns.")
    else:
        get_vif(df_copy)


def plot_histogram(df, col_name: str):
    """Plots two distribution plots - an histogram and a 
    emprical cumulative distribution - of the selected variable.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    df_copy = df.copy()
    median_val = df_copy[col_name].median()

    axis_format = plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    axis_format_prcnt = plt.FuncFormatter(lambda x, loc: f"{x:,.1%}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    hist = sns.histplot(df_copy[col_name], kde=True, ax=axes[0])
    hist.get_yaxis().set_major_formatter(axis_format)
    hist.axvline(median_val, linewidth=4, linestyle="--",
                 color='black')
    hist.set_title(f"Median '{col_name}' is {median_val:,}")

    ecd = sns.ecdfplot(df_copy[col_name], ax=axes[1])
    ecd.get_yaxis().set_major_formatter(axis_format_prcnt)
    ecd.set_title(f"Cumulative Distribution of Records by '{col_name}")

    plt.suptitle(f"Distribution of Records by '{col_name}'", size=16)
    plt.show()
