import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("Starting Phase 5: Building the AI Recommendation Engine...\n")

# 1. Load the Master Dataset
try:
    df = pd.read_csv('student_performance_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'student_performance_data.csv'.")
    exit()

# 2. Simulate Resource Ratings (1 to 5 stars, 0 means not tried)
np.random.seed(42)
resources = ['Video_Crash_Course', 'Practice_Worksheets', '1_on_1_Tutoring', 'Flashcard_App', 'Study_Group']

# Create random ratings with 50% sparsity (meaning students have only tried half the tools on average)
ratings = np.random.randint(0, 6, size=(len(df), len(resources)))
mask = np.random.rand(len(df), len(resources)) < 0.5
ratings[mask] = 0

# Add the ratings to a separate DataFrame for clean math
ratings_df = pd.DataFrame(ratings, columns=resources)

# 3. The Core Algorithm: Cosine Similarity
# This compares every student to every other student based on what they liked/disliked
print("Calculating the student similarity matrix...")
similarity_matrix = cosine_similarity(ratings_df)

# 4. The Recommendation Function
def recommend_resources(student_id_target, num_recommendations=2):
    # Find the index of the student in the dataframe
    student_index = df.index[df['Student_ID'] == student_id_target].tolist()[0]
    
    print(f"\n--- Generating Recommendations for Student {student_id_target} ---")
    student_ratings = ratings_df.iloc[student_index]
    
    # What has this student NOT tried yet? (Rating == 0)
    unseen_resources = student_ratings[student_ratings == 0].index.tolist()
    
    if not unseen_resources:
        print("This student has already tried every available study resource!")
        return
        
    print("Currently Tried & Rated:")
    for res, rating in student_ratings[student_ratings > 0].items():
        print(f"  - {res}: {rating} Stars")
    
    # Find the top 10 most similar students (excluding themselves)
    similar_students_indices = np.argsort(similarity_matrix[student_index])[::-1][1:11]
    
    # Calculate predicted scores for unseen resources based on what similar peers thought
    recommendation_scores = {}
    for resource in unseen_resources:
        # How did the similar students rate this specific resource?
        similar_ratings = ratings_df.iloc[similar_students_indices][resource]
        
        # Average the ratings, ignoring students who haven't tried it (0)
        valid_ratings = similar_ratings[similar_ratings > 0]
        if len(valid_ratings) > 0:
            recommendation_scores[resource] = valid_ratings.mean()
        else:
            recommendation_scores[resource] = 0
            
    # Sort from highest predicted score to lowest
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {num_recommendations} Recommended Actions:")
    for res, score in sorted_recommendations[:num_recommendations]:
        if score > 0:
            print(f"  -> Try {res} (Predicted Match Score: {score:.1f} / 5.0)")
        else:
            print(f"  -> Try {res} (Not enough peer data to score)")

# 5. Live Test! 
# Let's test it on Student 1146 (from your previous dataset example) and one other random student
recommend_resources(1146)
recommend_resources(1999)

print("\nPhase 5 Complete! Full Pipeline Successfully Built.")