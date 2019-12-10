import pandas as pd
import numpy as np
import datetime

raw = pd.read_csv("../data/raw/Border_Crossing_Entry_Data.csv")
print "date read"
raw['Date'] = pd.to_datetime(raw['Date'])
print "date converted"
raw['year'] = raw['Date'].dt.year
print "year extracted"
raw['month'] = raw['Date'].dt.month
print "month extracted"

raw.to_csv('../data/interim/date_transferred.csv')
print "Done"

df = pd.read_csv("../data/processed/processed.csv")
df_GDP = pd.read_csv("../data/raw/GDP_by_state.csv")

GDPs = []
for idx, row in df.iterrows():
    state, year, month = row['State'], row['year'], row['month']
    quarter = "Q{}".format(month / 4 + 1)
    col_name = "{}:{}".format(year, quarter)
    try:
        row = df_GDP[df_GDP.state == state]
        GDPs.append(row[col_name].values[0])
    except KeyError:
        GDPs.append(0)

df['GDP'] = GDPs
df.to_csv("../data/processed/with_GDP.csv")
print "Done"



