import pandas as pd

try:
    df = pd.read_csv('/home/nikhilesh/Projects/NeSY-Imitation-Learning/train.csv')
    print("Unique action indices:", df['action'].unique())
    print("Value counts:\n", df['action'].value_counts())
except Exception as e:
    print(e)
