import pandas as pd

def load_data(file_path):
    """load the ratings dataset."""
    # https://grouplens.org/datasets/movielens/32m/
    return pd.read_csv(file_path)

def get_dataset_stats(df):
    """get the needed dataset statistics for preprocessing"""
    # number of users and movies and ratings
    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    num_ratings = len(df)

    # find missing values
    missing_user_ids = df['userId'].isnull().sum()
    missing_movie_ids = df['movieId'].isnull().sum()

    # find min and max values
    min_user_id = df['userId'].min()
    max_user_id = df['userId'].max()
    min_movie_id = df['movieId'].min()
    max_movie_id = df['movieId'].max()

    return {
        'num_users': num_users,
        'num_movies': num_movies,
        'min_user_id': min_user_id,
        'max_user_id': max_user_id,
        'min_movie_id': min_movie_id,
        'max_movie_id': max_movie_id,
        'num_ratings': num_ratings,
    }

def preprocess_data(df):
    """preprocess the dataset."""
    # make the user_ids start from 0
    df['userId'] = df['userId'] - 1

    # remap movie ids
    movie_id_map = {movie_id: idx for idx, movie_id in enumerate(df['movieId'].unique())}
    df['movieId'] = df['movieId'].map(movie_id_map)

    # remove timestamp column
    df = df.drop(columns=['timestamp'])

    return df

def save_data(df, file_path):
    """save the preprocessed dataset."""
    df.to_csv(file_path, index=False)

def run_preprocessing_pipeline(BIG_DATA_DIR):
    """run the preprocessing pipeline."""
    # load data
    file_path = f'{BIG_DATA_DIR}/ratings.csv' # edit this path
    df = load_data(file_path)

    # get dataset statistics
    stats = get_dataset_stats(df)
    print(f"Number of users: {stats['num_users']}")
    print(f"Number of movies: {stats['num_movies']}")
    print(f"User ID range: {stats['min_user_id']} to {stats['max_user_id']}")
    print(f"Movie ID range: {stats['min_movie_id']} to {stats['max_movie_id']}")
    print(f"Number of ratings: {stats['num_ratings']}")
    
    """
    Output:
    Number of users: 200948
    Number of movies: 84432
    User ID range: 1 to 200948
    Movie ID range: 1 to 292757
    Number of ratings: 32000204

    There are 200,948 users, 84,432 movies, and 32M ratings, with no missing values in the dataset.
    """
    
    # preprocess data
    df = preprocess_data(df)

    # save data
    save_path = f'{BIG_DATA_DIR}/ratings_preprocessed.csv' # edit this path
    save_data(df, save_path)