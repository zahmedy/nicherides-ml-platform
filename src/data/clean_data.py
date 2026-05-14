def drop_more_than_4_nans(df):
    # drop 4 or more NaN rows
    df = df[df.isna().sum(axis=1) < 4]

    return df

def replace_low_price_with_avg(df):
    df['price_mean'] = df.groupby(['make', 'year', 'model'])['price'].transform('mean')
    df.loc[df['price'] < df['price_mean'], 'price'] = round(df['price_mean'])
    df.drop(columns=['price_mean'], inplace=True)

    return df