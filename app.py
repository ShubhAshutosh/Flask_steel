from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS  # Import CORS


app = Flask(__name__)

# Allow only Vercel frontend
CORS(app, resources={r"/predict": {"origins": "https://steel-prop.vercel.app"}})


# Load the trained RandomForest models for each property
tensile_strength_model = joblib.load('Tensile_Strength_rf_model.pkl')
elongation_model = joblib.load('Elongation_rf_model.pkl')
reduction_area_model = joblib.load('Reduction_in_Area_rf_model.pkl')
proof_stress_model = joblib.load('0.2_Proof_Stress_rf_model.pkl')
ceq_model = joblib.load('Ceq_rf_model.pkl')

# Initialize the Flask app
#app = Flask(__name__)
#CORS(app)

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Step 1: Get data from the POST request
        data = request.get_json()

        # Step 2: Prepare the input features from the received data
        features = np.array([[data['C'], data['Si'], data['Mn'], data['P'], data['S'],
                              data['Ni'], data['Cr'], data['Mo'], data['Cu'], data['V'],
                              data['Al'], data['N'], data['Temperature']]])

        # Step 3: Get the property to predict from the request (e.g., Tensile Strength or Elongation)
        property_to_predict = data['property']

        # Select the appropriate model based on the property
        if property_to_predict == 'tensile_strength':
            model = tensile_strength_model
        elif property_to_predict == 'elongation':
            model = elongation_model
        elif property_to_predict == 'reduction_in_area':
            model = reduction_area_model
        elif property_to_predict == 'proof_stress':
            model = proof_stress_model
        elif property_to_predict == 'ceq':
            model = ceq_model
        else:
            return jsonify({'error': 'Invalid property requested'}), 400

        # Step 4: Make a prediction using the model
        prediction = model.predict(features)

        # Step 5: Return the prediction in JSON format
        return jsonify({'predicted_value': prediction[0]})

    except Exception as e:
        # Handle errors, e.g., if input data is missing or invalid
        return jsonify({'error': str(e)}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
