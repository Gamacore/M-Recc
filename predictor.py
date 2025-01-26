import pickle

# Path to the saved item-item model data
model_data_path = './big_data/item_item_saved_model_data.pkl'

# Load the model data
with open(model_data_path, 'rb') as f:
    model_data = pickle.load(f)

# Extract neighbors, averages, and deviations from the saved data
neighbors = model_data['neighbors']
averages = model_data['averages']
deviations = model_data['deviations']

def predict(user_id, movie_id, neighbors, averages, deviations):
    """Return a rating prediction for the given user and movie."""
    if not neighbors[user_id]: # if the user has no neighbors, return the user's average rating
        return averages[user_id]

    numerator = 0
    denominator = 0
    prediction = averages[user_id] # default to the user's average rating
    for neg_w, neighbor_id in neighbors[user_id]:
        try:
            numerator += -neg_w * deviations[neighbor_id][movie_id]
            denominator += abs(neg_w)
        except KeyError:
            pass # neighbor_id did not rate movie_id
        
        if denominator == 0:
            prediction = averages[user_id] # nothing better we can do to predict
        else:
            prediction = numerator / denominator + averages[user_id]
        prediction = min(5, prediction) # cap the prediction, since ratings are between 0.5 and 5
        prediction = max(0.5, prediction) # cap the prediction, since ratings are between 0.5 and 5

    return prediction

user_id = 42  # Replace with a real user ID
movie_id = 120  # Replace with a real movie ID
predicted_rating = predict(user_id, movie_id, neighbors, averages, deviations)
print(f"Predicted rating for User {user_id} on Movie {movie_id}: {predicted_rating}")