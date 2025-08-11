# 📧 Spam vs Non-Spam Mail Classification

A machine learning project that classifies emails as **Spam** or **Non-Spam (Ham)** using Natural Language Processing (NLP) techniques and text preprocessing.  
The model is trained on a labeled dataset of email messages and leverages feature extraction methods like **TF-IDF** for accurate email filtering.

---

## 🚀 Features
- Classifies emails into **Spam** or **Non-Spam** categories.
- Uses **TF-IDF Vectorization** for feature extraction.
- Implements multiple machine learning algorithms (e.g., Naive Bayes, Logistic Regression).
- Includes data preprocessing (stopword removal, tokenization, lowercase conversion).
- Achieves high accuracy and precision on test data.

---

## 📂 Project Structure
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks for training & testing
├── spam_classifier.py # Main training & prediction script
├── requirements.txt # Dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🛠️ Tech Stack
- **Python** 🐍
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib & Seaborn** (for data visualization)
- **NLTK** (for NLP preprocessing)

---

## 📊 Workflow
1. **Data Collection** – Import spam/ham email dataset.
2. **Data Cleaning** – Remove punctuation, stopwords, and perform tokenization.
3. **Feature Extraction** – Convert text into numerical form using **TF-IDF**.
4. **Model Training** – Train ML algorithms (e.g., Naive Bayes, Logistic Regression).
5. **Evaluation** – Measure performance using accuracy, precision, recall, and F1-score.
6. **Prediction** – Classify new email messages.

---

## 📈 Example Results
| Model                | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| Naive Bayes          | 97.5%    | 96%       | 98%    | 97%      |
| Logistic Regression  | 96.8%    | 95%       | 97%    | 96%      |

---

## 📦 Installation & Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/spam-classifier.git
   cd spam-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run training script:

bash
Copy
Edit
python spam_classifier.py
Predict for new text:

python
Copy
Edit
from spam_classifier import predict_spam
print(predict_spam("Congratulations! You've won a free iPhone."))
📜 Dataset
This project uses the SpamAssassin dataset or the SMS Spam Collection dataset from UCI Machine Learning Repository.
