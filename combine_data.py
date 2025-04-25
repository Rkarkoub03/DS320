import pandas as pd

# Load datasets
imdb_df = pd.read_csv("C:/Users/Raed Karkoub/Desktop/DS320/imdb_top_1000.csv")
rt_df = pd.read_csv("C:/Users/Raed Karkoub/Desktop/DS320/rotten_tomatoes_top_movies.csv")

# Prepare IMDB dataset
imdb_clean = imdb_df[['Series_Title', 'Released_Year', 'Overview', 'IMDB_Rating']].copy()
imdb_clean.rename(columns={
    'Series_Title': 'title',
    'Released_Year': 'year',
    'Overview': 'review_text',
    'IMDB_Rating': 'rating'
}, inplace=True)
imdb_clean['source'] = 'IMDB'

# Prepare Rotten Tomatoes dataset
rt_clean = rt_df[['title', 'year', 'synopsis', 'critic_score']].copy()
rt_clean.rename(columns={
    'synopsis': 'review_text',
    'critic_score': 'rating'
}, inplace=True)
rt_clean['source'] = 'RottenTomatoes'

# Drop rows with missing reviews
imdb_clean.dropna(subset=['review_text'], inplace=True)
rt_clean.dropna(subset=['review_text'], inplace=True)

# Combine the two datasets
combined_df = pd.concat([imdb_clean, rt_clean], ignore_index=True)

# OPTIONAL: Filter out rows with missing or bad ratings
combined_df.dropna(subset=['rating'], inplace=True)


# View final combined dataset
print(combined_df.head())

# Save combined dataframe to a CSV file
combined_df.to_csv('C:/Users/Raed Karkoub/Desktop/DS320/combined_movie_reviews.csv', index=False)

