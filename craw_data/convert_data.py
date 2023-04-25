import pandas as pd
df = pd.read_csv('./data/raw/btc.csv')
div =1000
for i in range(len(df)):
  df.at[i,'open'] =df.iloc[i]['open']/div
  df.at[i,'high'] =df.iloc[i]['high']/div
  df.at[i,'close'] =df.iloc[i]['close']/div
  df.at[i,'low'] =df.iloc[i]['low']/div
df.to_csv('./data/clean/btc.csv')