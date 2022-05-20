import glob
import os
import re

import numpy as np
import pandas as pd
from cleantext import clean

def cleaner(x):
    x = re.sub(r'https?://[^ ]+', '', x)  # Remove URLs
    x = re.sub(r'@[^ ]+', '', x)  # Remove usernames
    x = re.sub(r'#', '', x)  # Hashtags
    x = re.sub(r'([A-Za-z])\1{2,}', r'\1', x)  # Character normalization
    # Punctuation, special characters and numbers
    x = re.sub(r' 0 ', 'zero', x)
    # Punctuation, special characters and numbers
    x = re.sub(r'[^A-Za-z ]', '', x)
    x = x.lower().lstrip()
    return x

all_filenames = [i for i in glob.glob('./dataset/*.csv')]

combined_csv = pd.concat([pd.read_csv(f)for f in all_filenames])

combined_csv['tweet_clean'] = combined_csv['tweet'].map(
    lambda x: clean(x, 
    no_urls=True, 
    no_emails=True, 
    no_numbers=True, 
    no_digits=True, 
    no_currency_symbols=True,  
    no_emoji=True
    ))

combined_csv['tweet_clean'] = combined_csv['tweet_clean'].map(
    lambda x: cleaner(x))
# drop line without tweets
combined_csv.replace(to_replace=r'^\s*$', value=np.nan,
                     regex=True, inplace=True)
combined_csv.dropna(subset=['tweet_clean'], inplace=True)


if not os.path.exists("combined_csv.csv"):
    combined_csv.to_csv("combined_csv.csv", index=False, encoding='utf-8')


df = combined_csv[0:int(combined_csv.shape[0] / 2)]
df.to_csv('train.csv', index=False)


df = combined_csv[int(combined_csv.shape[0] / 2) + 1:int(combined_csv.shape[0] / 4 * 3)]
df.to_csv('val.csv', index=False)


df = combined_csv[int(combined_csv.shape[0] / 4 * 3) + 1:combined_csv.shape[0]]
df.to_csv('test.csv', index=False)
