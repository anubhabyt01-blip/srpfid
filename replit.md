# Software Reliability Prediction Model

## Overview

This platform predicts software defect proneness (software reliability) using machine learning models optimized for imbalanced datasets. The system consists of a vanilla HTML/CSS/JavaScript frontend and a Python Flask backend with integrated ML engine.

The application enables researchers to upload defect datasets, configure and train ML models with various imbalance handling techniques (SMOTE, ADASYN, etc.), and evaluate model performance with comprehensive metrics and explainability features.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes

**October 17, 2025**: Complete stack conversion
- Converted from TypeScript/React/Vite to HTML/CSS/JavaScript frontend
- Replaced Node.js/Express backend with Python Flask
- Migrated from PostgreSQL to SQLite database
- Removed all Node.js dependencies and TypeScript configuration
- Created vanilla JavaScript frontend with clean, responsive design
- All functionality preserved in simpler technology stack

## System Architecture

### Frontend Architecture

**Technology Stack:**
- Pure HTML5 templates with Jinja2 templating
- Vanilla CSS with responsive design (no framework dependencies)
- Vanilla JavaScript for client-side logic
- No build tools or bundlers required

**Design Pattern:**
- Server-side routing with Flask
- Component-like HTML structure
- Modern CSS Grid and Flexbox layouts
- Fetch API for backend communication
- Event-driven DOM manipulation

**Key Pages:**
- Dashboard (`/`): System health, statistics overview, recent models/experiments
- Data Upload (`/data-upload`): Dataset management and file upload interface
- Model Training (`/model-training`): ML model configuration and training orchestration
- Quantum Lab (`/quantum-lab`): Experimental quantum ML algorithms
- Federated Learning (`/federated-learning`): Distributed learning coordination
- Monitoring (`/monitoring`): Real-time metrics and system monitoring

### Backend Architecture

**Flask Application** (`app.py`):
- Python Flask web framework
- RESTful API endpoints
- File upload handling with Werkzeug
- JSON API responses
- CORS enabled for development
- SQLite database integration

**Process Architecture:**
- Single Python Flask application
- Direct integration with ML backend
- File-based storage for datasets and models
- SQLite database for metadata persistence

### Python ML Pipeline

**Core Architecture** (`ml_backend.py`):
The ML backend implements a comprehensive pipeline with these stages:

1. **Data Loading & Preprocessing:**
   - CSV/JSON dataset ingestion
   - Feature/target separation
   - Label encoding for categorical variables

2. **Imbalance Handling:**
   - SMOTE (Synthetic Minority Over-sampling)
   - ADASYN (Adaptive Synthetic Sampling)
   - BorderlineSMOTE
   - RandomUnderSampler
   - BalancedRandomForestClassifier

3. **Feature Engineering:**
   - Train/validation/test stratified splitting
   - StandardScaler for feature normalization
   - SelectKBest for feature selection
   - Optional RFE (Recursive Feature Elimination)

4. **Model Training:**
   - RandomForestClassifier
   - Support Vector Machines (SVC)
   - Multi-layer Perceptron (MLPClassifier)
   - XGBoost (gradient boosting)
   - LightGBM (optional)

5. **Hyperparameter Optimization:**
   - GridSearchCV with F1-score optimization
   - Cross-validation for robust evaluation
   - Stratified K-fold splitting

6. **Model Evaluation:**
   - Confusion matrix, Precision, Recall, F1-score
   - Matthews Correlation Coefficient (MCC)
   - ROC-AUC score, Precision-Recall curves

7. **Model Explainability:**
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature importance rankings

8. **Model Persistence:**
   - Pickle serialization
   - JSON export for configurations

### Data Storage

**SQLite Database** (`app_database.db`):
- Primary data store for application metadata
- Tables: users, datasets, models, quantum_experiments, rl_agents, federated_nodes, federated_jobs, nlp_analysis, monitoring_metrics
- File-based storage in project root
- Simple setup with no external dependencies

**File System Storage:**
- `uploads/`: User-uploaded CSV/JSON datasets
- `data/`: Example datasets (e.g., nasa_defect_dataset.csv)
- `models/`: Serialized trained models and artifacts
- `static/`: Frontend assets (CSS, JavaScript)
- `templates/`: HTML templates for Flask rendering

**Design Decision:**
SQLite chosen for simplicity and zero-configuration deployment. Suitable for research platform with moderate concurrent users. Can be migrated to PostgreSQL for production scale if needed.

## Technology Stack Summary

### Frontend
- HTML5
- CSS3 (responsive design)
- Vanilla JavaScript (ES6+)

### Backend
- Python 3.11
- Flask 3.0
- SQLite3

### ML Libraries
- **Core ML**: scikit-learn, imbalanced-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy, scipy
- **Explainability**: SHAP, LIME
- **Experimental**: Qiskit, PennyLane, Cirq (quantum ML)
- **RL**: stable-baselines3, Ray RLlib (optional)

## Project Structure

```
.
├── app.py                 # Flask application (main entry)
├── init_db.py            # Database initialization script
├── ml_backend.py         # Machine learning pipeline
├── requirements.txt      # Python dependencies
├── app_database.db       # SQLite database
├── templates/            # HTML templates
│   ├── index.html
│   ├── data-upload.html
│   ├── model-training.html
│   ├── quantum-lab.html
│   ├── federated-learning.html
│   └── monitoring.html
├── static/               # Static assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── api.js
│       ├── dashboard.js
│       ├── data-upload.js
│       ├── model-training.js
│       ├── quantum-lab.js
│       ├── federated-learning.js
│       └── monitoring.js
├── uploads/              # User uploaded files
└── data/                 # Sample datasets
```

## Running the Application

The Flask application runs on port 5000:
```bash
python3 app.py
```

Access the application at: `http://0.0.0.0:5000`

## Future Enhancements

- Add user authentication and authorization
- Implement production WSGI server (Gunicorn/Waitress)
- Add comprehensive test suite
- Implement real-time WebSocket updates
- Add data visualization charts
- Integrate blockchain features
- Deploy to production environment
