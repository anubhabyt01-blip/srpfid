# Software Reliability Prediction Model for Imbalanced Datasets

## Overview
This platform provides comprehensive tools for predicting software defect proneness using machine learning models optimized for imbalanced datasets. Built with a modern HTML/CSS/JavaScript frontend and Python Flask backend, it features dataset management, model training, and AI-powered insights.

## Key Features

### Core Functionality
- **Dataset Upload & Management**: Upload CSV/JSON datasets with automatic analysis
- **Model Training**: Train ML models with advanced imbalance handling techniques
- **Imbalance Handling**: SMOTE, ADASYN, BorderlineSMOTE, RandomUnderSampler
- **Model Evaluation**: Comprehensive metrics including F1-score, MCC, AUC-ROC
- **Quantum ML Lab**: Experimental quantum machine learning algorithms
- **Federated Learning**: Distributed learning coordination
- **Real-time Monitoring**: System health and performance metrics

### New Features (Latest Release)

#### 🤖 Gemini AI Integration
- **Dataset Insights**: Get AI-powered analysis of dataset quality and recommendations
- **Model Performance Analysis**: Receive intelligent feedback on trained models
- **Training Recommendations**: AI-generated suggestions for hyperparameters and preprocessing

#### 📊 Balanced Dataset Export
- **Export Functionality**: Convert imbalanced datasets to balanced versions
- **Multiple Techniques**: Support for SMOTE, ADASYN, BorderlineSMOTE, and RandomUndersampling
- **Distribution Analytics**: View before/after class distributions
- **CSV Export**: Download balanced datasets for external use

## Technology Stack

### Frontend
- **HTML5**: Semantic markup with Jinja2 templating
- **CSS3**: Responsive design with modern Grid and Flexbox
- **Vanilla JavaScript**: No framework dependencies, pure ES6+

### Backend
- **Python 3.11**: Core language
- **Flask 3.0**: Web framework
- **SQLite**: Database (production-ready, no external dependencies)

### Machine Learning
- **scikit-learn**: Core ML algorithms
- **imbalanced-learn**: Handling imbalanced datasets
- **XGBoost**: Gradient boosting
- **SHAP & LIME**: Model explainability (optional)

### AI Integration
- **Google Gemini AI**: Advanced dataset and model analysis
- **Pydantic**: Data validation

## Project Structure

```
.
├── app.py                    # Flask application (main entry)
├── ml_backend.py            # ML pipeline and model training
├── gemini_service.py        # Gemini AI integration
├── init_db.py               # Database initialization
├── requirements.txt         # Python dependencies
├── app_database.db          # SQLite database
│
├── templates/               # HTML templates
│   ├── index.html          # Dashboard
│   ├── data-upload.html    # Dataset upload
│   ├── model-training.html # Model training
│   ├── quantum-lab.html    # Quantum experiments
│   ├── federated-learning.html
│   └── monitoring.html
│
├── static/                  # Static assets
│   ├── css/
│   │   └── style.css       # Application styles
│   └── js/
│       ├── api.js          # API client
│       ├── dashboard.js
│       ├── data-upload.js
│       ├── model-training.js
│       ├── quantum-lab.js
│       ├── federated-learning.js
│       └── monitoring.js
│
├── uploads/                 # User uploaded datasets
└── data/                    # Sample datasets
```

## Setup Instructions

### Prerequisites
- Python 3.11+
- pip (Python package manager)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Database**
   ```bash
   python3 init_db.py
   ```

4. **(Optional) Set up Gemini AI**
   - Get API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
   - Set environment variable:
     ```bash
     export GEMINI_API_KEY="your-api-key-here"
     ```

5. **Run the Application**
   ```bash
   python3 app.py
   ```

6. **Access the Application**
   - Open browser to `http://localhost:5000`

## Usage Guide

### 1. Upload Dataset
- Navigate to **Data Upload**
- Fill in dataset name and description
- Select your CSV or JSON file
- Click "Upload Dataset"
- View automatic analysis and quality metrics

### 2. Get AI Insights (New!)
- After uploading, click **AI Insights** button
- Receive Gemini-powered analysis including:
  - Data quality assessment
  - Preprocessing recommendations
  - Feature engineering suggestions

### 3. Export Balanced Dataset (New!)
- Click **Export Balanced** on any dataset
- Choose sampling technique:
  1. SMOTE (Synthetic Minority Over-sampling)
  2. ADASYN (Adaptive Synthetic Sampling)
  3. BorderlineSMOTE
  4. RandomUndersample
- Download balanced CSV file

### 4. Train Model
- Navigate to **Model Training**
- Select dataset and algorithm
- Configure hyperparameters (optional)
- Train and evaluate model
- View comprehensive metrics

### 5. Explore Advanced Features
- **Quantum Lab**: Experimental quantum ML algorithms
- **Federated Learning**: Distributed model training
- **Monitoring**: Real-time system metrics

## API Endpoints

### Datasets
- `GET /api/datasets` - List all datasets
- `GET /api/datasets/<id>` - Get dataset details
- `POST /api/datasets/upload` - Upload new dataset
- `GET /api/datasets/<id>/gemini-insights` - Get AI insights (New!)
- `POST /api/datasets/<id>/export-balanced` - Export balanced dataset (New!)

### Models
- `GET /api/models` - List all models
- `POST /api/models/train` - Train new model
- `GET /api/models/<id>/gemini-analysis` - Get AI analysis (New!)

### Utilities
- `GET /api/health` - System health check
- `POST /api/training-recommendations` - Get AI training tips (New!)

## Configuration

### Environment Variables
- `PORT`: Server port (default: 5000)
- `GEMINI_API_KEY`: Google Gemini API key (optional)
- `DATABASE_URL`: PostgreSQL URL (optional, uses SQLite by default)

## Supported Algorithms

### Classification Algorithms
- Random Forest
- Support Vector Machines (SVM)
- Neural Networks (MLP)
- XGBoost
- Balanced Random Forest

### Imbalance Handling Techniques
- SMOTE (Synthetic Minority Over-sampling Technique)
- ADASYN (Adaptive Synthetic Sampling)
- BorderlineSMOTE
- Random Undersampling
- Balanced Random Forest

## Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **MCC**: Matthews Correlation Coefficient
- **AUC-ROC**: Area under ROC curve
- **Confusion Matrix**: Detailed classification results

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit changes (`git commit -m 'Add YourFeature'`)
4. Push to branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

MIT License - See LICENSE file for details

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [Google Gemini AI](https://ai.google.dev/)

## Support

For questions or issues, please open an issue on GitHub.

---

**Version**: 2.0.0  
**Last Updated**: October 2025  
**Stack**: HTML/CSS/JavaScript + Python Flask
