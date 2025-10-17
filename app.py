#!/usr/bin/env python3

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import os
import json
from datetime import datetime
from werkzeug.utils import secure_filename
import sqlite3
import uuid

from ml_backend import MLBackend
from gemini_service import analyze_dataset_with_gemini, analyze_model_results, generate_training_recommendations

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'json', 'txt'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ml_backend = MLBackend()

def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect('app_database.db')
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data-upload')
def data_upload():
    return render_template('data-upload.html')

@app.route('/model-training')
def model_training():
    return render_template('model-training.html')

@app.route('/quantum-lab')
def quantum_lab():
    return render_template('quantum-lab.html')

@app.route('/federated-learning')
def federated_learning():
    return render_template('federated-learning.html')

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'services': {
            'ml': True,
            'quantum': True,
            'rl': True,
            'blockchain': True,
            'nlp': True,
            'federated': True
        }
    })

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM datasets ORDER BY uploaded_at DESC')
        datasets = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(datasets)
    except Exception as e:
        print(f"Error fetching datasets: {e}")
        return jsonify([])

@app.route('/api/datasets/<dataset_id>', methods=['GET'])
def get_dataset(dataset_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        dataset = cur.fetchone()
        conn.close()
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        return jsonify(dict(dataset))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            analysis = ml_backend.analyze_dataset(filepath)
            
            name = request.form.get('name', filename)
            description = request.form.get('description', '')
            
            conn = get_db_connection()
            cur = conn.cursor()
            dataset_id = str(uuid.uuid4())
            cur.execute('''
                INSERT INTO datasets (id, name, description, file_path, row_count, column_count, features, target_column, data_quality)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                dataset_id, name, description, filepath,
                analysis.get('rowCount'), analysis.get('columnCount'),
                json.dumps(analysis.get('features', [])),
                analysis.get('suggestedTarget'),
                json.dumps(analysis.get('quality', {}))
            ))
            conn.commit()
            
            cur.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
            dataset = dict(cur.fetchone())
            conn.close()
            
            return jsonify(dataset)
                
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM models ORDER BY created_at DESC')
        models = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(models)
    except Exception as e:
        print(f"Error fetching models: {e}")
        return jsonify([])

@app.route('/api/models/train', methods=['POST'])
def train_model():
    try:
        data = request.json
        
        conn = get_db_connection()
        cur = conn.cursor()
        model_id = str(uuid.uuid4())
        cur.execute('''
            INSERT INTO models (id, name, algorithm, dataset_id, hyperparameters, training_status)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            model_id,
            data.get('name'),
            data.get('algorithm'),
            data.get('datasetId'),
            json.dumps(data.get('hyperparameters', {})),
            'training'
        ))
        conn.commit()
        
        cur.execute('SELECT * FROM models WHERE id = ?', (model_id,))
        model = dict(cur.fetchone())
        conn.close()
        
        return jsonify(model)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/quantum/experiments', methods=['GET'])
def get_quantum_experiments():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM quantum_experiments ORDER BY created_at DESC')
        experiments = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(experiments)
    except Exception as e:
        return jsonify([])

@app.route('/api/quantum/experiments', methods=['POST'])
def create_quantum_experiment():
    try:
        data = request.json
        
        conn = get_db_connection()
        cur = conn.cursor()
        exp_id = str(uuid.uuid4())
        cur.execute('''
            INSERT INTO quantum_experiments (id, name, algorithm, qubits, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            exp_id,
            data.get('name'),
            data.get('algorithm'),
            data.get('qubits', 4),
            'pending'
        ))
        conn.commit()
        
        cur.execute('SELECT * FROM quantum_experiments WHERE id = ?', (exp_id,))
        experiment = dict(cur.fetchone())
        conn.close()
        
        return jsonify(experiment)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rl/agents', methods=['GET'])
def get_rl_agents():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM rl_agents ORDER BY created_at DESC')
        agents = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(agents)
    except Exception as e:
        return jsonify([])

@app.route('/api/federated/nodes', methods=['GET'])
def get_federated_nodes():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM federated_nodes')
        nodes = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(nodes)
    except Exception as e:
        return jsonify([])

@app.route('/api/federated/jobs', methods=['GET'])
def get_federated_jobs():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM federated_jobs ORDER BY created_at DESC')
        jobs = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(jobs)
    except Exception as e:
        return jsonify([])

@app.route('/api/monitoring/metrics', methods=['GET'])
def get_monitoring_metrics():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        source = request.args.get('source')
        
        if source:
            cur.execute('SELECT * FROM monitoring_metrics WHERE source = ? ORDER BY timestamp DESC LIMIT 100', (source,))
        else:
            cur.execute('SELECT * FROM monitoring_metrics ORDER BY timestamp DESC LIMIT 100')
        
        metrics = [dict(row) for row in cur.fetchall()]
        conn.close()
        return jsonify(metrics)
    except Exception as e:
        return jsonify([])

@app.route('/api/datasets/<dataset_id>/export-balanced', methods=['POST'])
def export_balanced_dataset(dataset_id):
    try:
        data = request.json
        sampling_technique = data.get('samplingTechnique', 'smote')
        target_column = data.get('targetColumn', 'defects')
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        dataset = cur.fetchone()
        conn.close()
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        result = ml_backend.export_balanced_dataset(
            dataset['file_path'],
            target_column,
            sampling_technique
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>/gemini-insights', methods=['GET'])
def get_gemini_insights(dataset_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        dataset = cur.fetchone()
        conn.close()
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        dataset_info = {
            'rowCount': dataset['row_count'],
            'columnCount': dataset['column_count'],
            'features': json.loads(dataset['features']) if dataset['features'] else {},
            'quality': json.loads(dataset['data_quality']) if dataset['data_quality'] else {}
        }
        
        insights = analyze_dataset_with_gemini(dataset_info)
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/<model_id>/gemini-analysis', methods=['GET'])
def get_model_analysis(model_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM models WHERE id = ?', (model_id,))
        model = cur.fetchone()
        conn.close()
        
        if not model:
            return jsonify({'error': 'Model not found'}), 404
        
        model_metrics = {
            'algorithm': model['algorithm'],
            'accuracy': model['accuracy'] or 0,
            'precision': model['precision'] or 0,
            'recall': model['recall'] or 0,
            'f1Score': model['f1_score'] or 0,
            'mcc': model['mcc'] or 0
        }
        
        analysis = analyze_model_results(model_metrics)
        return jsonify({'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training-recommendations', methods=['POST'])
def get_training_recommendations():
    try:
        data = request.json
        dataset_id = data.get('datasetId')
        algorithm = data.get('algorithm', 'RandomForest')
        
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
        dataset = cur.fetchone()
        conn.close()
        
        if not dataset:
            return jsonify({'error': 'Dataset not found'}), 404
        
        dataset_info = {
            'rowCount': dataset['row_count'],
            'quality': json.loads(dataset['data_quality']) if dataset['data_quality'] else {}
        }
        
        recommendations = generate_training_recommendations(dataset_info, algorithm)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
