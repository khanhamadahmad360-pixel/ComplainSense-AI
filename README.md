# ComplainSense AI
## Customer Complaint Intent Classification System

### 1. Project Overview
ComplainSense AI is a machine learning-based application designed to automatically identify customer support messages and classify them into relevant intent categories.

The system helps businesses organize customer queries and improve customer support efficiency.

### 2. Objectives
- Automatically classify customer messages.
- Identify customer support intents.
- Reduce manual work for customer support teams.
- Demonstrate machine learning through a web application.

### 3. Dataset
The project uses the Bitext Customer Support Dataset.

- Dataset: Bitext Customer Support
- Total Records: 26,872
- Number of Intent Classes: 27
- Features: Customer instructions and intent labels.

### 4. Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Streamlit
- Joblib

### 5. Machine Learning Model
The system uses:
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression classification

The trained model is saved as:
models/complainsense_final.pkl

### 6. Model Performance
The final model achieved 99.74% accuracy on the held-out test dataset.

Note: Actual performance on new, real-world customer messages may vary.

### 7. Application Features
- Customer message input
- Automatic intent prediction
- Confidence score
- Prediction confidence indicator
- Human review recommendation for low-confidence predictions

### 8. Project Structure
- app.py: Streamlit application
- ComplainSense_AI.ipynb: Model development notebook
- requirements.txt: Required Python packages
- models/: Trained machine learning models
- data/: Project data folder
- Bitext_Customer_Support.csv: Dataset

### 9. How to Run the Application

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

### 10. Limitations
- The model supports only its 27 trained intent categories.
- Out-of-scope messages may be incorrectly classified.
- Confidence scores are not calibrated probabilities.

### 11. Future Improvements
- Out-of-scope message detection
- Improved prediction confidence
- Top-three intent predictions
- Integration with customer support systems

### 12. Conclusion
ComplainSense AI demonstrates how machine learning and natural language processing can help classify customer support messages and support more efficient customer service operations.