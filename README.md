# ASD-Prognosis
# 🧠 Autism Prediction System using Machine Learning

## 🚀 Overview

This project presents a machine learning-based system designed to estimate the likelihood of Autism Spectrum Disorder (ASD) using user-provided inputs. The model combines structured behavioral questionnaire responses with optional facial image analysis to generate predictions.

The system utilizes an optimized **XGBoost classifier**, enhanced through hyperparameter tuning and data balancing techniques, and is deployed via an interactive **Streamlit web application** for ease of use.

---

## 💻 Demo Video

🔗 https://drive.google.com/file/d/1YQWXFSXtlV8HbRmffKJMPws130cBUeLq/view?usp=sharing

---

## 🌟 Key Features

* 📝 **Behavioral Screening Module**
  A structured set of questions designed to capture ASD-related behavioral patterns.

* 🖼️ **Optional Facial Analysis**
  Incorporates basic image-based feature insights to support prediction (experimental feature).

* ⚙️ **Machine Learning Pipeline**
  Uses XGBoost for classification with optimized parameters.

* ⚖️ **Handling Imbalanced Data**
  Applies SMOTE to improve model performance on minority classes.

* 📊 **Interactive Visualization**
  Displays dataset insights and prediction outcomes in real time.

* 🌐 **Streamlit Interface**
  Clean and user-friendly web interface for seamless interaction.

---

## 🛠️ Installation & Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/autism-predictor.git
   cd autism-predictor
   ```

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset

* Based on an Autism Screening dataset containing demographic and behavioral attributes.

* Includes features such as:

  * Age, gender, ethnicity
  * Questionnaire responses
  * ASD diagnosis labels

* Preprocessing steps:

  * Handling missing values
  * Removing redundant columns
  * Encoding categorical variables

---

## 🧠 Model & Approach

* **Preprocessing:** Data cleaning and feature encoding
* **Balancing Technique:** SMOTE for addressing class imbalance
* **Model Used:** XGBoost Classifier
* **Optimization:** GridSearchCV for tuning hyperparameters
* **Evaluation:** Accuracy-based performance assessment

---

## 🎯 How to Use

1. Fill out the behavioral questionnaire
2. (Optional) Upload an image for additional analysis
3. Click on **"Predict ASD"**
4. View prediction results along with basic interpretation

---

## 🖼️ Sample Output

🔗 https://acesse.one/CLICK-TO-OPEN-THE-RESULTS

---

## 🚀 Future Scope

* 🔹 Integrate advanced deep learning models for facial feature extraction
* 🔹 Expand dataset diversity for improved generalization
* 🔹 Add detailed reports and insights for users
* 🔹 Deploy on cloud for broader accessibility

---

## 🤝 Contribution

Contributions are welcome!

* Fork the repository
* Create a feature branch
* Commit your changes
* Submit a pull request

---

## 📬 Contact
Linkedin: https://www.linkedin.com/in/gunika-kharbanda-877238360/ 

