import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Paths for the saved models
MODEL_RF_PATH = 'model_rf.joblib'
MODEL_LR_PATH = 'model_lr.joblib'

# Dictionary to store loaded models
models = {}

def load_models():
    """Load joblib models into memory if they exist."""
    if os.path.exists(MODEL_RF_PATH):
        try:
            models['rf'] = joblib.load(MODEL_RF_PATH)
            print(f" -> Modelo Random Forest cargado con éxito desde '{MODEL_RF_PATH}'")
        except Exception as e:
            print(f"Error al cargar Random Forest: {e}")
            
    if os.path.exists(MODEL_LR_PATH):
        try:
            models['lr'] = joblib.load(MODEL_LR_PATH)
            print(f" -> Modelo Regresión Logística cargado con éxito desde '{MODEL_LR_PATH}'")
        except Exception as e:
            print(f"Error al cargar Regresión Logística: {e}")

# Pre-load models when server starts
load_models()

@app.route('/')
def index():
    """Render the main index page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request and return results."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No se proporcionaron datos de entrada'}), 400
        
        # Determine which model to use (default to Random Forest)
        model_type = data.get('model', 'rf')
        pipeline = models.get(model_type)
        
        # Fallback if model files are not yet loaded
        if not pipeline:
            # Re-attempt loading in case they were generated post-start
            load_models()
            pipeline = models.get(model_type)
            if not pipeline:
                return jsonify({
                    'error': f'El modelo de tipo "{model_type}" no está disponible. Asegúrate de ejecutar train.py primero.'
                }), 500
        
        # Cast attributes cleanly to ensure exact scikit-learn compatibility
        features = {
            'Pclass': int(data.get('Pclass', 3)),
            'Sex': str(data.get('Sex', 'male')).lower(),
            'Age': float(data.get('Age', 28.0)),
            'SibSp': int(data.get('SibSp', 0)),
            'Parch': int(data.get('Parch', 0)),
            'Fare': float(data.get('Fare', 20.0)),
            'Embarked': str(data.get('Embarked', 'S')).upper()
        }
        
        # Create input DataFrame with correct column names and matching order
        input_df = pd.DataFrame([features])
        
        # Generate prediction (0 = Died, 1 = Survived)
        prediction = int(pipeline.predict(input_df)[0])
        
        # Try to get prediction probability if supported
        probability = None
        try:
            prob_array = pipeline.predict_proba(input_df)[0]
            probability = float(prob_array[prediction])
        except Exception:
            pass
            
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'model_used': 'Random Forest' if model_type == 'rf' else 'Regresión Logística'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en el procesamiento: {str(e)}'}), 500

if __name__ == '__main__':
    # Run server locally on port 5001
    print(f"Iniciando servidor local en http://127.0.0.1:5001")
    app.run(debug=True, host='127.0.0.1', port=5001)
