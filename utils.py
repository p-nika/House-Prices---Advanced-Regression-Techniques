import pandas as pd


def preprocess_df(df, train_features, cat_features):
    """
    preprocesses the input dataframe to be ready for inference
    """

    for col in cat_features:
        df[col] = df[col].astype('category')

    df['YearBuilt'] = 2025 - df['YearBuilt']
    df['total_area_1st_2nd_floor'] = df['1stFlrSF'] + df['2ndFlrSF']
    df.rename(columns={'YearBuilt':'Age'}, inplace=True)
    df['total_area_1st_2nd_floor_bsmt'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['BsmtUnfSF']
    df['bsmt_diff'] = df['TotalBsmtSF'] - df['BsmtUnfSF']

    return df[train_features]