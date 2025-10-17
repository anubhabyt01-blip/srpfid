#!/usr/bin/env python3

# Referenced from blueprint:python_gemini integration
import json
import logging
import os

# Gemini AI integration for dataset analysis and insights
try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = os.environ.get("GEMINI_API_KEY") is not None
    if HAS_GEMINI:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
except ImportError:
    HAS_GEMINI = False

def analyze_dataset_with_gemini(dataset_info):
    """Use Gemini to provide insights about dataset quality and recommendations"""
    if not HAS_GEMINI:
        return {"insights": "Gemini AI not configured", "recommendations": []}
    
    try:
        prompt = f"""
        Analyze this software defect dataset and provide insights:
        
        Dataset Statistics:
        - Rows: {dataset_info.get('rowCount', 'N/A')}
        - Columns: {dataset_info.get('columnCount', 'N/A')}
        - Missing Values: {dataset_info.get('quality', {}).get('missingValues', 0):.2%}
        - Imbalance Ratio: {dataset_info.get('quality', {}).get('imbalanceRatio', 0):.2f}
        - Outliers: {dataset_info.get('quality', {}).get('outliers', 0):.2%}
        
        Features: {', '.join(list(dataset_info.get('features', {}).keys())[:10])}
        
        Provide:
        1. Data quality assessment (2-3 sentences)
        2. Top 3 recommendations for improving model performance
        3. Suggested preprocessing steps
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        return {
            "insights": response.text or "No insights generated",
            "recommendations": extract_recommendations(response.text)
        }
    except Exception as e:
        logging.error(f"Gemini analysis error: {e}")
        return {"insights": f"Error: {str(e)}", "recommendations": []}

def extract_recommendations(text):
    """Extract recommendations from Gemini response"""
    recommendations = []
    lines = text.split('\n')
    for line in lines:
        if any(keyword in line.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
            recommendations.append(line.strip())
    return recommendations[:5]

def analyze_model_results(model_metrics):
    """Get Gemini insights on model performance"""
    if not HAS_GEMINI:
        return "Gemini AI not configured"
    
    try:
        prompt = f"""
        Analyze these ML model metrics for software defect prediction:
        
        - Algorithm: {model_metrics.get('algorithm', 'N/A')}
        - Accuracy: {model_metrics.get('accuracy', 0):.2%}
        - Precision: {model_metrics.get('precision', 0):.2%}
        - Recall: {model_metrics.get('recall', 0):.2%}
        - F1 Score: {model_metrics.get('f1Score', 0):.3f}
        - MCC: {model_metrics.get('mcc', 0):.3f}
        
        Provide:
        1. Performance assessment
        2. Which metrics need improvement?
        3. Suggestions for hyperparameter tuning
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        return response.text or "No analysis generated"
    except Exception as e:
        return f"Error: {str(e)}"

def generate_training_recommendations(dataset_info, algorithm):
    """Generate AI-powered training recommendations"""
    if not HAS_GEMINI:
        return ["Use SMOTE for balancing", "Apply feature scaling", "Use cross-validation"]
    
    try:
        prompt = f"""
        For a software defect prediction model using {algorithm}:
        
        Dataset: {dataset_info.get('rowCount')} samples, imbalance ratio {dataset_info.get('quality', {}).get('imbalanceRatio', 0.5):.2f}
        
        Suggest 5 specific hyperparameters and preprocessing techniques to optimize performance.
        Format as a numbered list.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=prompt
        )
        
        recommendations = []
        for line in response.text.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-')):
                recommendations.append(line.strip())
        
        return recommendations[:5] or ["Apply SMOTE", "Use grid search", "Feature selection"]
    except Exception as e:
        return ["Use SMOTE for balancing", "Apply feature scaling", "Use cross-validation"]
