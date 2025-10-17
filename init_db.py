#!/usr/bin/env python3

import os
import sqlite3

def init_sqlite_db():
    """Initialize SQLite database with required tables"""
    db_path = 'app_database.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'researcher'
        );

        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            file_path TEXT NOT NULL,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            row_count INTEGER,
            column_count INTEGER,
            features TEXT,
            target_column TEXT,
            data_quality TEXT,
            preprocessing_steps TEXT
        );

        CREATE TABLE IF NOT EXISTS models (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            dataset_id TEXT,
            hyperparameters TEXT,
            training_status TEXT DEFAULT 'pending',
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            mcc REAL,
            confusion_matrix TEXT,
            feature_importance TEXT,
            model_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trained_by TEXT
        );

        CREATE TABLE IF NOT EXISTS quantum_experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            qubits INTEGER,
            circuits TEXT,
            results TEXT,
            optimization_results TEXT,
            execution_time REAL,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        );

        CREATE TABLE IF NOT EXISTS rl_agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            algorithm TEXT NOT NULL,
            environment TEXT NOT NULL,
            hyperparameters TEXT,
            training_progress TEXT,
            performance TEXT,
            model_path TEXT,
            status TEXT DEFAULT 'training',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        );

        CREATE TABLE IF NOT EXISTS federated_nodes (
            id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL UNIQUE,
            public_key TEXT NOT NULL,
            status TEXT DEFAULT 'offline',
            last_seen TIMESTAMP,
            reputation REAL DEFAULT 0,
            compute_capacity TEXT,
            blockchain_address TEXT,
            stake_amount REAL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS federated_jobs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            model_type TEXT NOT NULL,
            global_model TEXT,
            rounds INTEGER DEFAULT 0,
            participating_nodes TEXT,
            aggregation_results TEXT,
            blockchain_hashes TEXT,
            status TEXT DEFAULT 'preparing',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT
        );

        CREATE TABLE IF NOT EXISTS nlp_analysis (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            document_type TEXT NOT NULL,
            content TEXT NOT NULL,
            extracted_features TEXT,
            sentiment REAL,
            complexity REAL,
            topics TEXT,
            entities TEXT,
            embeddings TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS monitoring_metrics (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            metadata TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    ''')
    
    conn.commit()
    conn.close()
    print(f"SQLite database initialized at {db_path}")

if __name__ == '__main__':
    init_sqlite_db()
