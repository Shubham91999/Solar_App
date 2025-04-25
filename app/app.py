import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import plotly.express as px
import plotly
import json
import pandas as pd


# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model (Solar energy prediction model)
model = pickle.load(open("solar_energy_model.pkl", "rb"))

# Load the dataset
file_path = 'Merged_Dataset.csv'  # Adjust this path accordingly
data = pd.read_csv(file_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get user input data for prediction
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    
    # Predict solar energy production
    prediction = model.predict(features)
    
    # Return result to front-end
    return render_template("index.html", prediction_text=f"Predicted Solar Energy: {prediction[0]} kWh")

# Define a route for the dashboard
@app.route("/dashboard")
def dashboard():
    # 1. Prepare the data for the "Annual Solar Energy Production by County" chart
    county_avg = data.groupby('county')['Solar_Energy2022'].mean().reset_index()
    county_avg_sorted = county_avg.sort_values(by='Solar_Energy2022', ascending=False)

    # 2. Prepare the data for the "Comparison of Features for Solar Energy Prediction" chart
    features = ['ghi_mean2022', 'dni_mean2022', 'dhi_mean2022', 'temp_mean2022', 'wind_mean2022', 'sp_mean2022']
    feature_data = data[features + ['Solar_Energy2022']].corr()['Solar_Energy2022'].drop('Solar_Energy2022')

    # 3. Prepare data for the "Solar Energy Production by Location" (Geo Map)
    map_data = data[['latitude', 'longitude', 'Solar_Energy2022']].dropna()

    # Convert the Plotly graphs to JSON format for rendering in the HTML template
    county_avg_json = json.dumps(county_avg_sorted.to_dict(orient='records'))
    feature_data_json = json.dumps(feature_data.to_dict())
    map_data_json = json.dumps(map_data.to_dict(orient='records'))

    return render_template('dashboard.html', 
                           county_avg=county_avg_json, 
                           feature_data=feature_data_json, 
                           map_data=map_data_json)








if __name__ == "__main__":
    app.run(debug=True)
