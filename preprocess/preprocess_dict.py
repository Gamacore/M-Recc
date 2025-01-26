import pandas as pd
import pickle

def load_data(file_path):
    """load the ratings dataset. by default, we are using the shrunk dataset."""
    # https://grouplens.org/datasets/movielens/32m/
    return pd.read_csv(file_path)

def train_test_split(df, train_test_split=0.8):
    """split the dataset into train and test sets."""
    
    # shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # find the split index
    split_index = int(len(df) * train_test_split)
    
    # split the dataset
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    del df # delete the original dataframe to save memory
    
    return train_df, test_df

def update_dicts(row, user2movie, movie2user, usermovie2rating):
    """update the user2movie, movie2user, and usermovie2rating dictionaries."""
    user_id = int(row['userId'])
    movie_id = int(row['movieId'])
    rating = row['rating']

    # update user2movie
    if user_id not in user2movie:
        user2movie[user_id] = [movie_id]
    else:
        user2movie[user_id].append(movie_id)

    # update movie2user
    if movie_id not in movie2user:
        movie2user[movie_id] = [user_id]
    else:
        movie2user[movie_id].append(user_id)

    # update usermovie2rating
    usermovie = (user_id, movie_id)
    usermovie2rating[usermovie] = rating

def save_dict(dict_, file_path):
    """save the dictionary to a file."""
    with open(file_path, 'wb') as f:
        pickle.dump(dict_, f)

def run_dictify(BIG_DATA_DIR):
    """dictify the dataset. can change the train_test_split as needed."""
    # load the dataset
    file_path = f'{BIG_DATA_DIR}/ratings_shrunk.csv'
    df = load_data(file_path)
    
    # split the dataset
    train_df, test_df = train_test_split(df)

    user2movie = {}
    movie2user = {}
    usermovie2rating = {}
    user2movie_test = {}
    movie2user_test = {}
    usermovie2rating_test = {}

    # update the dictionaries
    train_df.apply(update_dicts, axis=1, args=(user2movie, movie2user, usermovie2rating))
    test_df.apply(update_dicts, axis=1, args=(user2movie_test, movie2user_test, usermovie2rating_test))

    # save the dictionaries
    save_dict(user2movie, f'{BIG_DATA_DIR}/user2movie.pkl')
    save_dict(movie2user, f'{BIG_DATA_DIR}/movie2user.pkl')
    save_dict(usermovie2rating, f'{BIG_DATA_DIR}/usermovie2rating.pkl')
    save_dict(usermovie2rating_test, f'{BIG_DATA_DIR}/usermovie2rating_test.pkl')