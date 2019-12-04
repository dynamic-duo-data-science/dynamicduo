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

