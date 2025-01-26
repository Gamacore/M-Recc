# **🎬 Movie Recommendation Systems**

## **📝 Description**

This project implements various **Movie Recommendation Systems**. Currently, it supports **user-based collaborative filtering** and **item-based collaborative filtering** to recommend movies based on user preferences and historical data.

### ✨ Features:
- 🔗 **User-Based Collaborative Filtering**: Recommends movies based on similarities between users.
- 🎥 **Item-Based Collaborative Filtering**: Recommends movies based on similarities between movies.
- 🛠️ **Customizable Dataset**: Easily add your own ratings to test the system.
- 📊 **Scalable**: Works seamlessly with large datasets like [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/).

### **Planned Features**

Here are the planned features to expand the functionality of this project:

1. **🧮 Advanced Matrix Factorization:**
   - Incorporate methods such as Bayesian Matrix Factorization and Probabilistic Matrix Factorization for better handling of sparse datasets.

2. **🚀 Neural Network Models:**
   - Implement deep learning approaches like Autoencoders and Residual Learning for enhanced recommendation quality.

3. **🔗 Restricted Boltzmann Machines (RBMs):**
   - Include RBM-based collaborative filtering for better latent representation of user-item interactions.

4. **⚡ Scalable Implementations with Spark:**
   - Optimize matrix factorization algorithms for distributed environments using Apache Spark.

5. **📊 Bayesian Approaches:**
   - Extend the system with Bayesian sampling and ranking techniques to improve ranking accuracy for recommendations.

6. **📚 Hybrid Recommendation System:**
   - Combine collaborative filtering with content-based methods to create a robust hybrid recommendation model.

7. **🌐 Cloud-Based Deployment:**
   - Set up recommendation pipelines in AWS or other cloud environments for real-world scalability.

8. **📈 Real-World Applications:**
   - Expand the project to make predictions for real-world datasets and evaluate system performance on unseen, live data.

Let me know if you’d like further elaboration on any of these planned features! 🚀
---

## **📊 Results**

### **Performance Metrics**:

- **User-Based Collaborative Filtering**:
  - 🟢 **Train MSE**: `0.5571904374226911`
  - 🟢 **Test MSE**: `0.6216365099123221`

- **Item-Based Collaborative Filtering**:
  - 🟢 **Train MSE**: `0.4709482615323109`
  - 🟢 **Test MSE**: `0.5650172300648622`

## **🛠️ Installation**

### Dependencies
The project uses Python 3.7+ and the following libraries:
- `numpy`
- `pandas`
- `sortedcontainers`
- `pickle`

Install them using:
```bash
pip install -r requirements.txt
```

---

## **🚀 Usage**

### **0. Dataset**
The project uses the [MovieLens 32M dataset](https://grouplens.org/datasets/movielens/32m/). Download the dataset and place it in the `big_data` directory.

### **1. Preprocess the Dataset**
Prepare the MovieLens dataset or your own data:
```bash
python run_preprocessing.py
```
This script runs three scripts:

Script 1:
- Cleans the dataset.
- Remaps `userId` and `movieId` to contiguous indices.
- Saves the processed dataset as `ratings_preprocessed.csv`.

Script 2:
- Shrinks the dataset to focus on the most active users and movies.

Script 3:
- Creates dictionaries for user-to-movie, movie-to-user, and user-movie ratings.

### **2. Train Collaborative Filtering Models**
- **User-Based Collaborative Filtering**:
  ```bash
  python user_user.py
  ```
- **Item-Based Collaborative Filtering**:
  ```bash
  python item_item.py
  ```

### **3. Generate Recommendations**
This is under development. Currently, the `recommend.py` script can be modified to generate recommendations for users.

---

## **📂 Project Structure**

```plaintext
├── big_data/                  # Directory for storing data
│   ├── ratings.csv            # Original dataset
│   ├── ratings_preprocessed.csv  # Preprocessed dataset
│   ├── ratings_shrunk.csv     # Reduced dataset
│   ├── user2movie.pkl         # User-to-movie mapping
│   ├── movie2user.pkl         # Movie-to-user mapping
│   ├── usermovie2rating.pkl   # Train ratings dictionary
│   ├── usermovie2rating_test.pkl  # Test ratings dictionary
│   └── [model results]
├── preprocess.py              # Data preprocessing
├── shrink.py                  # Dataset shrinking
├── user_based_cf.py           # User-based collaborative filtering
├── item_based_cf.py           # Item-based collaborative filtering
├── predictor.py               # Functions for recommendations
├── requirements.txt           # Dependency list
└── README.md                  # Documentation
```