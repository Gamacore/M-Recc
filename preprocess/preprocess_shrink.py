import pandas as pd
from collections import Counter

def load_data(file_path):
    """load the preprocessed dataset."""
    return pd.read_csv(file_path)

def remap_ids(df, new_user_ids, new_movie_ids):
    """remap the user and movie ids."""
    # remap user ids
    user_id_map = {user_id: idx for idx, user_id in enumerate(new_user_ids)}
    df['userId'] = df['userId'].map(user_id_map)
    
    # remap movie ids
    movie_id_map = {movie_id: idx for idx, movie_id in enumerate(new_movie_ids)}
    df['movieId'] = df['movieId'].map(movie_id_map)
    
    return df

def save_data(df, file_path):
    """save the remapped dataset."""
    df.to_csv(file_path, index=False)

def run_shrinking(BIG_DATA_DIR, user_threshold=100, movie_threshold=100):
    """shrink the dataset. you can adjust the thresholds user_threshold and movie_threshold as needed."""
    file_path = f'{BIG_DATA_DIR}/ratings_preprocessed.csv' # edit this path
    df = load_data(file_path)

    # find the most active users
    user_counts = Counter(df['userId'])
    most_active_users = [user_id for user_id, _ in user_counts.most_common(user_threshold)]
    
    # find the most active movies
    movie_counts = Counter(df['movieId'])
    most_active_movies = [movie_id for movie_id, _ in movie_counts.most_common(movie_threshold)]
    
    # filter the dataset
    df = df[df['userId'].isin(most_active_users) & df['movieId'].isin(most_active_movies)].copy()

    # remap the user and movie ids
    df = remap_ids(df, most_active_users, most_active_movies)
    
    print(f"Number of users: {len(most_active_users)}")
    print(f"Number of movies: {len(most_active_movies)}")
    print(f"Number of ratings: {len(df)}")


    file_path = f'{BIG_DATA_DIR}/ratings_shrunk.csv' # edit this path
    save_data(df, file_path)