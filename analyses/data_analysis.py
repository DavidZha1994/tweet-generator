import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('../dataset/combined_Musk_tweets_cleaned.txt', sep="\n", header=None)
data.columns = ["txt"]
word_lengths = data['txt'].apply(lambda x: len(x.split()))
bins = word_lengths.value_counts()
bins = bins.sort_index()

print(f"avg_len={np.mean(bins)}")

plt.bar(range(len(bins)), bins.values)
plt.xlabel("tweet length [words]")
plt.ylabel("occurrences")
plt.xticks(range(0, len(bins), 10), range(1, len(bins) + 1, 10))
plt.show()
