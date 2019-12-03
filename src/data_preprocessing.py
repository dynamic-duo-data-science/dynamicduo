import pandas as pd
import numpy as np
import datetime

raw = pd.read_csv("../data/raw/Border_Crossing_Entry_Data.csv")
raw['Date'] = pd.to_datetime(raw['Date'])

coordinates_X, coordinates_Y = [], []
# month_offset = []
SMALLEST_YEAR = 2016
for idx, row in raw.iterrows():
    coordinate_str = row['Location'].split('POINT (')[1].split(' ')
    co_x, co_y = coordinate_str[0], coordinate_str[1].split(')')[0]
    coordinates_X.append(co_x)
    coordinates_Y.append(co_y)
    agg_month = (raw['Date'].dt.year - SMALLEST_YEAR) * 12 + raw['Date'].dt.month
    # month_offset.append(agg_month)

raw['location_x'] = coordinates_X
raw['location_y'] = coordinates_Y
raw['year'] = raw['Date'].dt.year
raw['month'] = raw['Date'].dt.month

raw.to_csv('../data/interim/interim.csv')
print "Done"

