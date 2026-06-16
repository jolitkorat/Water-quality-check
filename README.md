# 💧 Water Quality Prediction System

An AI-powered machine learning system that predicts whether water is safe for drinking based on chemical properties.

## 🚀 Features

- **Data Preprocessing** — Missing value imputation, feature scaling
- **Exploratory Data Analysis** — Interactive charts (heatmaps, histograms, box plots)
- **5 ML Models** — Logistic Regression, Random Forest, Decision Tree, SVM, KNN
- **Auto Model Selection** — Automatically picks the best-performing model
- **Real-time Prediction** — Input water parameters and get instant results
- **Downloadable Reports** — Export prediction results
- **Professional UI** — Ocean-themed Streamlit dashboard with glassmorphism design

## 📁 Project Structure

```
water-quality-prediction/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── water_potability.csv      # Original dataset
├── dataset/
│   └── water_potability.csv  # Organized dataset copy
├── src/
│   ├── __init__.py           # Package init
│   ├── data_preprocessing.py # Data loading & cleaning
│   ├── eda.py                # Visualization functions
│   ├── model_training.py     # ML model training & evaluation
│   └── prediction.py         # Prediction & report generation
├── models/
│   └── best_model.pkl        # Saved best model (auto-generated)
├── assets/                   # Static assets
└── notebooks/                # Jupyter notebooks
```

## 🧪 Dataset

The dataset contains **3,277 water samples** with the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| pH | Acid/base balance | 0-14 |
| Hardness | Mineral content | mg/L |
| Solids | Total dissolved solids | ppm |
| Chloramines | Disinfectant level | ppm |
| Sulfate | Sulfate concentration | mg/L |
| Conductivity | Electrical conductivity | μS/cm |
| Organic_carbon | Total organic carbon | ppm |
| Trihalomethanes | THM concentration | μg/L |
| Turbidity | Water clarity | NTU |
| **Potability** | **Target (0=Unsafe, 1=Safe)** | Binary |

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd water-quality-prediction

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

## 🤖 ML Models

| Model | Type | Key Parameters |
|-------|------|---------------|
| Logistic Regression | Linear | max_iter=1000, C=1.0 |
| Random Forest | Ensemble | n_estimators=200, max_depth=10 |
| Decision Tree | Tree | max_depth=8, min_samples_split=5 |
| SVM | Kernel | kernel=rbf, C=1.0 |
| KNN | Instance-based | n_neighbors=7, weights=distance |

## 🚀 Deployment

### Streamlit Cloud
1. Push code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set main file path to `app.py`
5. Click **Deploy**

### Render
1. Push code to GitHub
2. Create a new **Web Service** on [Render](https://render.com)
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Deploy

## 📊 Tech Stack

- **Python 3.x** — Core language
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Machine learning
- **Plotly** — Interactive visualizations
- **Matplotlib & Seaborn** — Static plots
- **Streamlit** — Web dashboard
- **Joblib** — Model serialization

## 📝 License

This project is for educational purposes. MIT License.
