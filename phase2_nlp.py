import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

print("Starting Phase 2: Training NLP Sentiment Classifier...\n")

# 1. Load the dataset
try:
    df = pd.read_csv('student_performance_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'student_performance_data.csv'. Make sure you ran generate_data.py first!")
    exit()

# 2. Define our target: Nuanced Sentiment Logic
def assign_sentiment(row):
    if row['Final_Score'] >= 80:
        return 1  # High scorers always get Positive (1)
    elif row['Final_Score'] < 60:
        return 0  # Low scorers always get Negative (0)
    else:
        # The Middle Ground (Scores between 60 and 79)
        # Give a positive overall label if they show good effort and decent history
        if row['Hours_Studied'] >= 5.0 and row['Attendance_Pct'] >= 75.0 and row['Past_Grade'] >= 65.0:
            return 1
        else:
            return 0

# Apply our custom function to every row in the dataframe
df['Is_Positive_Note'] = df.apply(assign_sentiment, axis=1)

X = df['Teacher_Notes']
y = df['Is_Positive_Note']

# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Text Vectorization (Converting Words to Numbers)
# We remove 'english' stop words (like 'the', 'is', 'and', 'but') because they don't add sentiment meaning
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)  # Only transform the test set!

# 5. Train a Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 6. Make Predictions and Evaluate
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"NLP Model Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix:")
print("(True Negatives, False Positives)")
print("(False Negatives, True Positives)")
print(confusion_matrix(y_test, y_pred))

# 7. Let's test the model on brand new, complex unseen text!
test_notes = [
    "Consistently submits assignments on time and shows excellent analytical thinking.", # Pure Positive
    "Frequently distracted during lectures and needs significant improvement in fundamentals.", # Pure Negative
    "Active participant and eager to learn but struggling with algebra concepts." # Mixed Bag!
]

print("\n--- Live Text Classification Test ---")
test_vec = vectorizer.transform(test_notes)
predictions = model.predict(test_vec)

# We can also ask the model *how confident* it is in its prediction
probabilities = model.predict_proba(test_vec)

for i, (note, pred) in enumerate(zip(test_notes, predictions)):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = np.max(probabilities[i]) * 100 if 'np' in globals() else max(probabilities[i]) * 100
    print(f"Note: '{note}'")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence:.1f}%)\n")
    
print("Phase 2 Complete!")