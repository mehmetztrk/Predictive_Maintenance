
# 🛠️ Predictive Maintenance using Tree-based Models

This project builds a supervised machine learning pipeline to predict equipment failures using historical sensor data.  
It includes complete steps from preprocessing to training and evaluation — all in a production-style modular setup.

---

## 📁 Project Structure

```
predictive_maintenance/
│
├── data/                 # Raw and processed CSV files
├── notebooks/            # EDA notebook
├── scripts/              # Feature engineering, training, evaluation scripts
├── models/               # Saved models (pkl)
├── output/               # Evaluation outputs (plots & predictions)
├── main.py               # One-click pipeline entry
├── README.md             # Project description
└── requirements.txt      # Required packages
```

---

## 🔍 Problem Statement

Can we predict if a machine is going to fail based on sensor readings like torque, temperature, and speed?

This project answers that by:
- Engineering features from raw IoT-style telemetry data
- Training tree-based models (Decision Tree, Random Forest)
- Evaluating performance with accuracy, confusion matrix, and ROC-AUC
- Visualizing model insights (feature importance)

---

## 📊 Dataset

- Source: AI4I 2020 Predictive Maintenance Dataset
- Size: ~10,000 samples
- Features: Temperature, Speed, Torque, Tool wear, etc.
- Label: `Machine failure` (0: No, 1: Yes)

---

## ⚙️ Technologies Used

- Python 3.8  
- pandas, NumPy, matplotlib, seaborn  
- scikit-learn, joblib  
- ROC, AUC, confusion matrix, feature importance  

---

## 🚀 How to Run

1. Clone the repository  
2. Create a virtual environment & install requirements:

```bash
pip install -r requirements.txt
```

3. Run the full pipeline:

```bash
python main.py
```

This will:
- Preprocess the data  
- Train models  
- Generate evaluation outputs (in `/output`)

---

## 📈 Results

- **Random Forest Accuracy**: 99.9%  
- **Decision Tree Accuracy**: 99.8%  
- Both models performed well; Random Forest showed better precision with fewer false positives.

🔽 Outputs:
- `output/roc_curve.png`
- `output/feature_importance.png`
- `output/predictions.csv`

---

## 🧠 Author

**Mehmet Ozturk**  
Feel free to connect: [GitHub](https://github.com/mehmetztrk)

---

## 💬 Feedback

Open to suggestions, ideas, and collaboration!  
Follow the repo to stay updated with future ML projects. 🚀
