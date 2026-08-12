import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)
num_students = 2000

print("Generating synthetic student dataset...")

# 1. Generate Core Numeric Features
hours_studied = np.random.uniform(1.0, 10.0, num_students)  # 1 to 10 hours per week
attendance_pct = np.random.uniform(60.0, 100.0, num_students)  # 60% to 100% attendance
past_grades = np.random.uniform(40.0, 100.0, num_students)  # Previous exam scores (out of 100)

# 2. Generate Target for Regression (Final Exam Score)
noise = np.random.normal(0, 5, num_students)
final_score = (0.4 * past_grades) + (3.5 * hours_studied) + (0.3 * attendance_pct) + noise
final_score = np.clip(final_score, 0, 100)  # Keep scores bounded between 0 and 100

# 3. Generate Unstructured Text Data (Teacher Notes for NLP)
pool_hours_pos = ["Shows excellent analytical thinking", "Demonstrates great understanding of core concepts"]
pool_att_pos = ["Consistently submits assignments on time", "Active participant and eager to learn", "Highly engaged in class discussions"]
pool_past_pos = ["Outstanding performance this semester"]

pool_hours_neg = ["Struggling with algebra concepts", "Needs significant improvement in fundamentals"]
pool_att_neg = ["Inconsistent attendance affecting performance", "Finds it difficult to keep up with the pace"]
pool_past_neg = ["Needs significant improvement in fundamentals", "Frequently distracted during lectures"]

teacher_notes = []

for i in range(num_students):
    score = final_score[i]
    hours = hours_studied[i]
    attendance = attendance_pct[i]
    past = past_grades[i]
    
    good_traits = []
    bad_traits = []
    
    # 1. Evaluate Every Feature Independently
    if hours > 5.0:
        good_traits.append(np.random.choice(pool_hours_pos))
    elif hours < 2.0:
        bad_traits.append(np.random.choice(pool_hours_neg))
        
    if attendance > 80.0:
        good_traits.append(np.random.choice(pool_att_pos))
    elif attendance < 65.0:
        bad_traits.append(np.random.choice(pool_att_neg))
        
    if past > 75.0:
        good_traits.append(np.random.choice(pool_past_pos))
    elif past < 55.0:
        bad_traits.append(np.random.choice(pool_past_neg))
        
    # 2. Construct the Sentence Dynamically
    note = ""
    
    # Case A: Student has BOTH positive and negative traits (The "Mixed Bag")
    if len(good_traits) > 0 and len(bad_traits) > 0:
        good = np.random.choice(good_traits)
        bad = np.random.choice(bad_traits)
        
        # Randomize the sentence structure
        if np.random.rand() > 0.5:
            note = f"{good} but {bad.lower()}."
        else:
            note = f"{bad} but {good.lower()}."
            
    # Case B: Only Positive Traits
    elif len(good_traits) > 0 and len(bad_traits) == 0:
        if len(good_traits) >= 2:
            np.random.shuffle(good_traits)
            note = f"{good_traits[0]} and {good_traits[1].lower()}."
        else:
            note = f"{good_traits[0]}."
            
    # Case C: Only Negative Traits
    elif len(bad_traits) > 0 and len(good_traits) == 0:
        if len(bad_traits) >= 2:
            np.random.shuffle(bad_traits)
            note = f"{bad_traits[0]} and {bad_traits[1].lower()}."
        else:
            note = f"{bad_traits[0]}."
            
    # Case D: The "True Average" (Scores are completely in the middle)
    else:
        all_pos = pool_hours_pos + pool_att_pos + pool_past_pos
        all_neg = pool_hours_neg + pool_att_neg + pool_past_neg
        
        if score >= 65:
            # Leans positive, but might have a small critique
            note = np.random.choice([
                f"{np.random.choice(all_pos)}.",
                f"{np.random.choice(all_pos)} but {np.random.choice(all_neg).lower()}."
            ], p=[0.7, 0.3])
        else:
            # Leans negative, but might have a small compliment
            note = np.random.choice([
                f"{np.random.choice(all_neg)}.",
                f"{np.random.choice(all_neg)} but {np.random.choice(all_pos).lower()}."
            ], p=[0.7, 0.3])
            
    # Capitalize the very first letter just in case, and save the note
    note = note[0].upper() + note[1:]
    teacher_notes.append(note)

# 4. Generate Handwritten Grade Labels (For Image Classification Alignment)
handwritten_assignment_grade = np.random.randint(0, 10, num_students)

# 5. Generate Target for Imbalanced Multi-Class Classification (Top 5%, Average, Dropout 10%)
# Calculate thresholds based on the generated final scores
top_5_threshold = np.percentile(final_score, 95)
bottom_10_threshold = np.percentile(final_score, 10)

# Assign categories based on thresholds using np.select
conditions = [
    (final_score >= top_5_threshold),
    (final_score <= bottom_10_threshold)
]
choices = ['Top 5%', 'Dropout Risk']
student_category = np.select(conditions, choices, default='Average')

# Combine everything into a Pandas DataFrame
df = pd.DataFrame({
    'Student_ID': range(1001, 1001 + num_students),
    'Hours_Studied': np.round(hours_studied, 1),    
    'Attendance_Pct': np.round(attendance_pct, 1),
    'Past_Grade': np.round(past_grades, 1),
    'Teacher_Notes': teacher_notes,
    'Handwritten_Grade_Label': handwritten_assignment_grade,
    'Student_Category': student_category,
    'Final_Score': np.round(final_score, 1)
})

# Save to CSV
df.to_csv('student_performance_data.csv', index=False)
print(f"Success! Saved {num_students} student records to 'student_performance_data.csv'.")

# Print a summary of our new Multi-Class distribution
print("\n--- Class Distribution for Phase 4 ---")
print(df['Student_Category'].value_counts(normalize=True) * 100)