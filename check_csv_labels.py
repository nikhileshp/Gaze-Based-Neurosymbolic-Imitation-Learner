import pandas as pd
df = pd.read_csv('data/seaquest/test.csv')
print("Action counts in test.csv:")
print(df['action'].value_counts())
