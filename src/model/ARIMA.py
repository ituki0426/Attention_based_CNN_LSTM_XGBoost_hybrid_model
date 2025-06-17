import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import metrics

data_ch = pd.read_csv('./data/raw/601988.SH.csv')
data_google = pd.read_csv('./data/raw/GOOG.csv')

data_ch.index = pd.to_datetime(data_ch['trade_date'], format='%Y%m%d') 
data_google.index = pd.to_datetime(data_google['Date'],format='%m/%d/%Y')
