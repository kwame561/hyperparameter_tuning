#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def describe_numeric_data(df):
    """takes a pandas data frame and describes
    with basic statistics, as long as all the 
    columns are nueric
    """
    df_copy = df.copy()
    df_res = df_copy.describe().T
    df_res['dtypes'] = df_copy.dtypes
    df_res['NULLs'] = df_copy.isna().sum()
    print(f"data : {df_res.shape[0]:,} rows x {df_res.shape[1]:,} columns")
    return df_res