import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'data', 'cleaned_train_data.csv')
if not os.path.exists(train_path):
    train_path = os.path.join(base_dir, 'raw_backup', 'train_data.csv')
df = pd.read_csv(train_path, encoding='utf-8')

print('Total Samples:', len(df))
print('\nClass Distribution (Stars 1-5):')
dist = df['star_rating'].value_counts().sort_index()
for star, count in dist.items():
    pct = (count / len(df)) * 100
    print(f'  {star} Stars: {count:,} samples ({pct:.2f}%)')

print('\nSample 1 Preview:')
print('Rating:', df.iloc[0]['star_rating'])
print('Text:', str(df.iloc[0]['review_body'])[:100], '...')