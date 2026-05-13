

def filter_columns(df):
    features = ["price","make","model","year","body_type",
                "fuel_type","engine_volume","mileage",
                "engine_cylinders","transmission",
                "drivetrain","color"]
    df = df[features]

    return df

def filter_bad_rows(df):
    # remove bad data
    df = df[(df['price'] > 500) & (df['price'] < 500000)]
    df = df[(df['year'] > 1970) & (df['year'] < 2028)]
    df = df[(df['mileage'] > 1) & (df['mileage'] < 500000)]
    df = df[(df['engine_volume'] >= 1) & (df['engine_volume'] < 7)]
    df = df[(df['engine_cylinders'] >= 3) & (df['engine_cylinders'] < 17)]

    return df

def drop_more_than_4_nans(df):
    # drop 4 or more NaN rows
    df = df[df.isna().sum(axis=1) < 4]

    return df

