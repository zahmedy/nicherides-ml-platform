def drop_more_than_4_nans(df):
    # drop 4 or more NaN rows
    df = df[df.isna().sum(axis=1) < 4]

    return df