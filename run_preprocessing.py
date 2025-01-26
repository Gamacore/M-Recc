import os
from preprocess import run_preprocessing_pipeline, run_shrinking, run_dictify

def main():
    BIG_DATA_DIR = './big_data'

    # Define the path to the preprocessed file
    preprocessed_file = f'{BIG_DATA_DIR}/ratings_preprocessed.csv'
    shrunk_file = f'{BIG_DATA_DIR}/ratings_shrunk.csv'
    
    # Check if the file exists
    if os.path.exists(preprocessed_file):
        print(f"{preprocessed_file} already exists. Skipping preprocessing.")
    else:
        print(f"{preprocessed_file} not found. Running preprocessing pipeline...")
        run_preprocessing_pipeline(BIG_DATA_DIR)
        print("Preprocessing complete.")

    # Check if the file exists
    if os.path.exists(shrunk_file):
        print(f"{shrunk_file} already exists. Skipping shrinking.")
    else:
        print(f"{shrunk_file} not found. Running shrinking pipeline...")
        run_shrinking(BIG_DATA_DIR, 500, 500)
        print("Shrinking complete.")
    
    # Run the dictify pipeline
    if (os.path.exists(f'{BIG_DATA_DIR}/user2movie.pkl') and os.path.exists(f'{BIG_DATA_DIR}/movie2user.pkl') and os.path.exists(f'{BIG_DATA_DIR}/usermovie2rating.pkl') and os.path.exists(f'{BIG_DATA_DIR}/usermovie2rating_test.pkl')):
        print("Dictionaries already exist. Skipping dictify.") # if one of the dictionaries is missing, rerun the dictify pipeline
    else:
        print("Dictionaries not found. Running dictify pipeline...")
        run_dictify(BIG_DATA_DIR)
        print("Dictify complete.")

if __name__ == '__main__':
    main()