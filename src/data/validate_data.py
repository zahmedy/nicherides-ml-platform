from datetime import date

def filter_columns(df):
    features = ["price","make","model","year","body_type",
                "fuel_type","engine_volume","mileage",
                "engine_cylinders","transmission",
                "drivetrain","color"]
    df = df[features]

    return df

def filter_bad_rows(df):
    curr_year = date.today().year
    # remove bad data
    df = df[(df['price'] > 1) & (df['price'] < 300000)]
    df = df[(df['year'] > 1970) & (df['year'] <= curr_year)]
    df = df[(df['mileage'] > 1) & (df['mileage'] < 500000)]
    df = df[(df['engine_volume'] >= 1) & (df['engine_volume'] < 7)]
    df = df[(df['engine_cylinders'] >= 3) & (df['engine_cylinders'] < 17)]

    return df