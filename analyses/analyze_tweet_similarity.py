from Levenshtein import distance
import numpy as np
import pandas as pd
from pathlib import Path

path = Path('../sample_generated_tweets')
for df_path in path.iterdir():

    if not df_path.name.endswith(".csv"):
        continue

    file_path = r"../dataset/combined_Musk_tweets_cleaned.txt"
    df = pd.read_csv(df_path)
    generated_tweets = df['output']
    dists = []

    with open(file_path) as f:
        for generated_tweet in generated_tweets.iteritems():
            min_dist = np.inf
            most_similar_original = ""

            for original_tweet in f:
                d = distance(original_tweet, generated_tweet[1])
                if d < min_dist:
                    min_dist = d
                    most_similar_original = original_tweet

            f.seek(0)   # reset pointer for file
            dists.append(min_dist)
            print(f"similarity: {min_dist}\n generated: {generated_tweet[1]} \n original:{most_similar_original}")

    print(f"{df_path.name}: {np.mean(dists)}")
