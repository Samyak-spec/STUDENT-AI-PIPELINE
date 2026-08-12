# Student AI/ML Multi-Phase Project

A hands-on machine learning and artificial intelligence project that builds a complete student-performance analysis system through multiple AI/ML techniques.

The project progresses from traditional machine learning to Natural Language Processing, Computer Vision, advanced classification, and recommendation systems.

---

## 🚀 Project Overview

This project contains five major AI/ML phases:

1. **Numerical Regression** – Predict student final exam scores.
2. **NLP Sentiment Classification** – Analyze teacher notes and classify them as positive or negative.
3. **Computer Vision CNN** – Recognize handwritten digits using the MNIST dataset.
4. **Multi-Class Classification** – Categorize students using XGBoost while handling class imbalance.
5. **Recommendation Engine** – Recommend study resources using student similarity and collaborative filtering.

A synthetic dataset containing **2,000 student records** is generated for the project.

---

## 📂 Project Structure

```text
Student-AI-ML-Project/
│
├── Student_data.py
├── student_performance_data.csv
│
├── phase1_regression.py
├── phase2_nlp.py
├── phase3_vision.py
├── phase4_droupout.py
├── phase5_recommend.py
│
├── test1.py
├── requirements.txt
└── README.md
```

---

# 🧠 Phase 1 – Numerical Regression

### File

`phase1_regression.py`

This phase uses **Linear Regression** to predict a student's final score.

### Features

* Hours Studied
* Attendance Percentage
* Past Grade

### Target

* Final Score

The data is divided into training and testing sets, numeric features are standardized using `StandardScaler`, and a Linear Regression model is trained. The model is evaluated using **Mean Absolute Error (MAE)** and **R² score**.

### Run

```bash
python phase1_regression.py
```

---

# 📝 Phase 2 – NLP Sentiment Classification

### File

`phase2_nlp.py`

This phase analyzes teacher comments using **Natural Language Processing**.

Teacher notes are converted from text into numerical features using **TF-IDF**, and a **Multinomial Naive Bayes** classifier predicts whether a note is positive or negative.

### Techniques Used

* TF-IDF Vectorization
* Stop-word removal
* Multinomial Naive Bayes
* Accuracy
* Confusion Matrix
* Prediction confidence

The script also tests the trained model on new teacher comments that were not part of the training dataset.

### Run

```bash
python phase2_nlp.py
```

---

# 👁️ Phase 3 – Computer Vision CNN

### File

`phase3_vision.py`

This phase builds a **Convolutional Neural Network (CNN)** using PyTorch to recognize handwritten digits from the MNIST dataset.

The CNN contains:

* Conv2D layer: 1 → 32 feature maps
* Conv2D layer: 32 → 64 feature maps
* ReLU activation
* Max Pooling
* Dropout
* Fully Connected layers
* Cross Entropy Loss
* Adam optimizer

The two convolution layers detect increasingly complex visual patterns such as edges, curves, and loops.

The model is trained for 5 epochs using batches of images and then evaluated on unseen MNIST test data.

### Run

```bash
python phase3_vision.py
```

The first run downloads the MNIST dataset automatically.

---

# 📊 Phase 4 – Multi-Class Student Classification

### File

`phase4_droupout.py`

This phase classifies students into three categories:

* **Top 5%**
* **Average**
* **Dropout Risk**

The project uses:

* StandardScaler
* TF-IDF
* LabelEncoder
* XGBoost
* Sample weighting
* Classification Report
* Confusion Matrix

The dataset contains both numerical features and teacher notes, so the preprocessing pipeline handles both types of data.

XGBoost is configured with constraints such as limited tree depth, learning rate, subsampling, column sampling, and gamma-based pruning to reduce overfitting.

Class imbalance is handled using balanced sample weights.

### Run

```bash
python phase4_droupout.py
```

---

# 🎯 Phase 5 – AI Recommendation Engine

### File

`phase5_recommend.py`

This phase implements a **user-based collaborative filtering recommendation system**.

The system simulates student ratings for different study resources:

* Video Crash Course
* Practice Worksheets
* 1-on-1 Tutoring
* Flashcard App
* Study Group

A rating of `0` represents a resource that the student has not tried.

### How it works

1. Calculate similarity between students using **Cosine Similarity**.
2. Find the 10 students most similar to the target student.
3. Find resources the target student has not tried.
4. Look at how similar students rated those resources.
5. Ignore missing ratings.
6. Calculate the average rating.
7. Rank resources by predicted score.
8. Recommend the highest-scoring resources.

The recommendation logic is implemented using the similarity matrix and the ratings of the top 10 similar students.

### Run

```bash
python phase5_recommend.py
```

---

# 🗃️ Dataset

### File

`Student_data.py`

This script generates the project's synthetic dataset.

It creates:

* 2,000 students
* Hours studied
* Attendance percentage
* Past grades
* Teacher notes
* Handwritten grade labels
* Student category
* Final score

The final dataset is saved as:

```text
student_performance_data.csv
```

The generator also creates the student categories using percentile-based thresholds for the top 5% and bottom 10%.

### Generate the dataset

```bash
python Student_data.py
```

Run this before the phases if `student_performance_data.csv` does not already exist.

---

# 🛠️ Technologies Used

| Technology        | Purpose                                      |
| ----------------- | -------------------------------------------- |
| Python            | Main programming language                    |
| Pandas            | Data manipulation                            |
| NumPy             | Numerical computation                        |
| Scikit-learn      | Traditional ML, preprocessing and evaluation |
| PyTorch           | Neural network training                      |
| Torchvision       | MNIST computer vision dataset                |
| Matplotlib        | Visualization                                |
| XGBoost           | Advanced classification                      |
| TF-IDF            | Text feature extraction                      |
| Cosine Similarity | Student similarity calculation               |

---

# 📦 Installation

## 1. Install Python

Install Python 3.10+ from the official Python website.

Check your installation:

```bash
python --version
```

---

## 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

If using PowerShell:

```bash
.\venv\Scripts\Activate.ps1
```

---

## 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

## 4. Install Required Libraries

```bash
pip install pandas numpy scikit-learn matplotlib xgboost torch torchvision
```

---

## 5. Generate the Dataset

```bash
python Student_data.py
```

This creates:

```text
student_performance_data.csv
```

---

# ▶️ Running the Complete Project

Run the phases in order:

```bash
python Student_data.py
python phase1_regression.py
python phase2_nlp.py
python phase3_vision.py
python phase4_droupout.py
python phase5_recommend.py
```

---

# 📈 Machine Learning Concepts Demonstrated

This project demonstrates:

* Train/Test Split
* Feature Scaling
* Linear Regression
* MAE
* R² Score
* TF-IDF
* Naive Bayes
* Confusion Matrix
* Neural Networks
* CNN
* Convolution
* Max Pooling
* ReLU
* Dropout
* Cross Entropy Loss
* Adam Optimizer
* XGBoost
* Label Encoding
* Class Imbalance
* Sample Weighting
* Cosine Similarity
* Collaborative Filtering
* Recommendation Systems

---

# 🎓 Learning Objective

The goal of this project is to demonstrate how different types of data can be processed using different AI/ML techniques.

```text
Numerical Data
      ↓
Linear Regression
      ↓
Score Prediction

Text Data
      ↓
TF-IDF + Naive Bayes
      ↓
Sentiment Classification

Image Data
      ↓
CNN
      ↓
Digit Recognition

Mixed Data
      ↓
XGBoost
      ↓
Student Classification

Student Preferences
      ↓
Cosine Similarity
      ↓
Recommendation Engine
```

---

# 👨‍💻 Author

**Samyak Srivastava**

B.Tech – Electronics & Communication Engineering
Minor in Computer Science

This project was developed as a practical exploration of machine learning, NLP, computer vision, classification, and recommendation systems.
