import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

print("Starting Phase 4: Multi-Class Imbalanced Classification with XGBoost...\n")

# 1. Load the Master Dataset
try:
    df = pd.read_csv('student_performance_data.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find 'student_performance_data.csv'.")
    exit()

# 2. Define Features and Target
numeric_features = ['Hours_Studied', 'Attendance_Pct', 'Past_Grade', 'Handwritten_Grade_Label']
text_feature = 'Teacher_Notes'

X = df[numeric_features + [text_feature]]
y_raw = df['Student_Category']

# XGBOOST REQUIREMENT: XGBoost cannot read text labels like "Average". It requires 0, 1, 2.
# LabelEncoder translates our string categories into math buckets.
le = LabelEncoder()
y = le.fit_transform(y_raw)

# 3. Train/Test Split (Still using stratify!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} students, Testing on {len(X_test)} students.\n")

# 4. Build the Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('text', TfidfVectorizer(stop_words='english'), text_feature)
    ])

# 5. Build the XGBoost Pipeline (WITH ANTI-OVERFITTING LEASHES)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        max_depth=3,              # LEASH 1: Keep decision trees shallow (prevent memorization)
        learning_rate=0.1,        # LEASH 2: Force it to learn slowly
        subsample=0.8,            # LEASH 3: Train each tree on a random 80% of the students
        colsample_bytree=0.8,     # LEASH 4: Train each tree on a random 80% of the columns (Great for text!)
        gamma=2,                  # LEASH 5: Pruning - Only make a split if it significantly improves the tree
        random_state=42,
        eval_metric='mlogloss'
    ))
])

# 6. Calculate Sample Weights (The XGBoost alternative to class_weight='balanced')
# This calculates the exact mathematical multiplier needed to boost the voices of the rare students.
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 7. Train the Model
print("Training the XGBoost Classifier. This may take a few seconds...\n")
# We pass the sample_weights directly into the classifier step of the pipeline
model.fit(X_train, y_train, classifier__sample_weight=sample_weights)

# 8. Evaluate the Predictions
y_pred = model.predict(X_test)

# Convert the 0,1,2 predictions back into English ("Average", "Top 5%") so we can read the report
y_test_english = le.inverse_transform(y_test)
y_pred_english = le.inverse_transform(y_pred)

print("--- Classification Report ---")
print(classification_report(y_test_english, y_pred_english))

print("--- Confusion Matrix ---")
labels = le.classes_
cm = confusion_matrix(y_test_english, y_pred_english, labels=labels)
cm_df = pd.DataFrame(cm, index=[f"True {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
print(cm_df)

print("\nPhase 4 Complete!")