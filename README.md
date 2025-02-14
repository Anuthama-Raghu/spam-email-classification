# Spam Email Classification using Machine Learning

## Project Overview
This project is a **machine learning-based spam classifier** built as part of my **Data Science coursework at UMass Amherst**. The goal is to develop a model that accurately distinguishes spam emails from legitimate ones using **NLP techniques and classification algorithms**.

## Technologies Used
- **Python**: Pandas, NumPy, Scikit-learn
- **NLP**: TF-IDF Vectorization, Text Preprocessing
- **Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

## Dataset
- **Source**: [UCI Machine Learning Repository](https://www.kaggle.com/uciml/sms-spam-collection-dataset)
- **Size**: ~50,000 email samples (or actual count from your dataset)
- **Classes**: Spam (1) and Not Spam (0)

## Methodology
1. **Data Preprocessing**  
   - Removed stopwords, punctuation, and special characters  
   - Converted text to lowercase and tokenized  
   - Applied **TF-IDF vectorization** for feature extraction  

2. **Model Training & Comparison**  
   - Trained multiple classifiers (Logistic Regression, Random Forest, XGBoost)  
   - Optimized hyperparameters using **GridSearchCV**  

3. **Model Evaluation**  
   - **Achieved 95% accuracy** with XGBoost  
   - Evaluated using **Confusion Matrix, Precision, Recall, and F1-score**  

## Results
- The **XGBoost model outperformed** other classifiers with:
  - **Accuracy**: 95%
  - **Precision**: 94%
  - **Recall**: 93%
  - **F1-Score**: 94%
- The model successfully classified spam emails with **minimal false positives**.

## Key Takeaways
  - **TF-IDF vectorization** significantly improved text feature extraction.  
  - **XGBoost performed the best** due to its ability to capture complex patterns.  
  - Preprocessing techniques like **stemming and stopword removal** enhanced accuracy.  

## Next Steps
  - Deploy model using Flask or Streamlit for real-world usage.  
  - Explore **deep learning** techniques (LSTMs, BERT) for further accuracy improvement.  

---

## 🔗 Connect with Me
📧 **Email**: anuthamarb@gmail.com  
🔗 **LinkedIn**: [linkedin.com/in/anuthamaraghu](https://www.linkedin.com/in/anuthamaraghu/)  
📂 **Portfolio**: Coming soon!
