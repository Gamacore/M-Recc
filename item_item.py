import pickle
import os
import numpy as np
from sortedcontainers import SortedList

def load_data(file_path):
    """load the data."""
    if not os.path.exists(f'{file_path}/user2movie.pkl') or \
        not os.path.exists(f'{file_path}/movie2user.pkl') or \
        not os.path.exists(f'{file_path}/usermovie2rating.pkl') or \
        not os.path.exists(f'{file_path}/usermovie2rating_test.pkl'):
        raise FileNotFoundError(f"{file_path} not found.")
    with open(f'{file_path}/user2movie.pkl', 'rb') as f:
        user2movie = pickle.load(f)

    with open(f'{file_path}/movie2user.pkl', 'rb') as f:
        movie2user = pickle.load(f)

    with open(f'{file_path}/usermovie2rating.pkl', 'rb') as f:
        usermovie2rating = pickle.load(f)

    with open(f'{file_path}/usermovie2rating_test.pkl', 'rb') as f:
        usermovie2rating_test = pickle.load(f)

    return user2movie, movie2user, usermovie2rating, usermovie2rating_test

def data_stats(user2movie, movie2user, usermovie2rating):
    """

    """
    n1 = np.max(list(user2movie.keys())) # number of users
    n2 = np.max([u for (u, m), r in usermovie2rating.items()])
    m1 = np.max(list(movie2user.keys())) # number of movies
    m2 = np.max([m for (u, m), r in usermovie2rating.items()])
    N = max(n1, n2) + 1
    M = max(m1, m2) + 1
    print(f"Number of users: {N}")
    print(f"Number of movies: {M}")

    if N > 10000 or M > 2000:
        print("This dataset is very big and will consume a lot of memory.")
        print("Do you want to continue?")
        print("Enter 'y' to continue or 'n' to exit")
        response = input()
        if response != 'y':
            exit()

    # # print some stats
    # print(f"Number of users: {len(user2movie)}")
    # print(f"Number of movies: {len(movie2user)}")
    # print(f"Number of ratings: {len(usermovie2rating)}")
    # print(f"Number of test ratings: {len(usermovie2rating_test)}")

    return N, M

def get_neighbors(movie_id, user2movie, movie2user, usermovie2rating, M, neighbors, averages, deviations, K=25, limit=5):

    # find the movies the user has watched
    users_i = movie2user[movie_id]
    users_i_set = set(users_i)

    # calculate the average and deviation
    ratings_i = {user:usermovie2rating[(user, movie_id)] for user in users_i} # user's ratings for movies
    avg_i = np.mean(list(ratings_i.values())) # user's average rating
    dev_i = {user:(rating - avg_i) for user, rating in ratings_i.items()} # user's deviation from average
    dev_i_values = np.array(list(dev_i.values())) # convert to numpy array
    sigma_i = np.sqrt(dev_i_values.dot(dev_i_values)) # user's deviation magnitude, square root of the sum of squares

    # save the average and deviation for later use
    averages.append(avg_i)
    deviations.append(dev_i)

    # loop over other users
    sorted_neighbors = SortedList() # since we only want the top K neighbors, we can truncate the list to 25 items
    for neighbor_id in range(M):
        if neighbor_id != movie_id:
            users_neighbor_id = movie2user[neighbor_id]
            users_neighbor_id_set = set(users_neighbor_id)
            common_movies = (users_i_set & users_neighbor_id_set) # intersection of the two sets
            if len(common_movies) > limit:
                # calculate the average and deviation
                ratings_neighbor_id = {user:usermovie2rating[(user, neighbor_id)] for user in users_neighbor_id}
                avg_neighbor_id = np.mean(list(ratings_neighbor_id.values()))
                dev_neighbor_id = {user:(rating - avg_neighbor_id) for user, rating in ratings_neighbor_id.items()}
                dev_neighbor_id_values = np.array(list(dev_neighbor_id.values()))
                sigma_neighbor_id = np.sqrt(dev_neighbor_id_values.dot(dev_neighbor_id_values))

                if sigma_i == 0 or sigma_neighbor_id == 0: # to avoid division by zero
                    continue

                # calculate the correlation coefficient
                numerator = sum(dev_i[m]*dev_neighbor_id[m] for m in common_movies)
                w_ij = numerator / (sigma_i * sigma_neighbor_id)

                sorted_neighbors.add((-w_ij, neighbor_id)) # sorts by descending order of w_ij
                if len(sorted_neighbors) > K:
                    del sorted_neighbors[-1]
    
    # store the neighbors
    neighbors.append(sorted_neighbors)
    save_model_data(neighbors, averages, deviations)

def predict(user_id, movie_id, neighbors, averages, deviations):
    """Return a rating prediction for the given user and movie."""
    if not neighbors[movie_id]: # if the user has no neighbors, return the user's average rating
        return averages[movie_id]

    numerator = 0
    denominator = 0
    prediction = averages[movie_id] # default to the user's average rating
    for neg_w, neighbor_id in neighbors[movie_id]:
        try:
            numerator += -neg_w * deviations[neighbor_id][user_id]
            denominator += abs(neg_w)
        except KeyError:
            pass # neighbor_id did not rate movie_id
        
        if denominator == 0:
            prediction = averages[movie_id] # nothing better we can do to predict
        else:
            prediction = numerator / denominator + averages[movie_id]
        prediction = min(5, prediction) # cap the prediction, since ratings are between 0.5 and 5
        prediction = max(0.5, prediction) # cap the prediction, since ratings are between 0.5 and 5

    return prediction

def mse(p, t):
    return np.mean((p - t) ** 2)

def save_model_data(neighbors, averages, deviations, path='./big_data/item_item_saved_model_data.pkl'):
    import pickle
    data = {
        'neighbors': neighbors,
        'averages': averages,
        'deviations': deviations
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_model_data(path='model_data.pkl'):
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['neighbors'], data['averages'], data['deviations']

def save_all(
    neighbors,
    averages,
    deviations,
    train_predictions,
    train_targets,
    test_predictions,
    test_targets,
    train_mse,
    test_mse,
    path='./big_data/all_results.pkl'
):
    all_data = {
        'neighbors': neighbors,
        'averages': averages,
        'deviations': deviations,
        'train_predictions': train_predictions,
        'train_targets': train_targets,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'train_mse': train_mse,
        'test_mse': test_mse,
    }
    with open(path, 'wb') as f:
        pickle.dump(all_data, f)


def main():
    BIG_DATA_DIR = './big_data'

    # load the data
    user2movie, movie2user, usermovie2rating, usermovie2rating_test = load_data(BIG_DATA_DIR)

    # get the dataset statistics
    N, M = data_stats(user2movie, movie2user, usermovie2rating)
    
    neighbors = [] # store neighbors
    averages = [] # each user's average rating
    deviations = [] # each user's deviation

    # get the neighbors
    for i in range(M):
        get_neighbors(movie_id=i, user2movie=user2movie, movie2user=movie2user, usermovie2rating=usermovie2rating, M=M, neighbors=neighbors, averages=averages, deviations=deviations)    
        if i % 1 == 0:
            print(i)
    
    train_predictions = np.zeros(len(usermovie2rating))
    train_targets = np.zeros(len(usermovie2rating))
    for idx, ((user_id, movie_id), target) in enumerate(usermovie2rating.items()):
        prediction = predict(user_id, movie_id, neighbors, averages, deviations)
        train_predictions[idx] = prediction
        train_targets[idx] = target

    test_predictions = np.zeros(len(usermovie2rating_test))
    test_targets = np.zeros(len(usermovie2rating_test))
    for idx, ((user_id, movie_id), target) in enumerate(usermovie2rating_test.items()):
        prediction = predict(user_id, movie_id, neighbors, averages, deviations)
        test_predictions[idx] = prediction
        test_targets[idx] = target

    train_mse = mse(train_predictions, train_targets)
    test_mse = mse(test_predictions, test_targets)

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    save_all(
        neighbors,
        averages,
        deviations,
        train_predictions,
        train_targets,
        test_predictions,
        test_targets,
        train_mse,
        test_mse,
        path=f'{BIG_DATA_DIR}/all_item_item_results.pkl'
    )
    
if __name__ == '__main__':
    main()