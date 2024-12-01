from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the model
with open('addclick1.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form inputs
            daily_time_spent = float(request.form['daily_time_spent'])
            age = int(request.form['age'])
            area_income = float(request.form['area_income'])
            daily_internet_usage = float(request.form['daily_internet_usage'])
            male = int(request.form['male'])  # Male (1) or Female (0)

            # Prepare the input array
            input_data = np.array([[daily_time_spent, age, area_income, daily_internet_usage, male]])
            
            # Ensure the number of features matches the model's training
            if input_data.shape[1] != model.n_features_in_:
                raise ValueError(f"Expected {model.n_features_in_} features, but got {input_data.shape[1]}.")

            # Make a prediction
            prediction = model.predict(input_data)
            output = prediction[0]

            # Prepare prediction text
            prediction_text = f'Likelihood of clicking on the ad: {"Yes" if output == 1 else "No"}'
            return render_template('result.html', prediction_text=prediction_text)
        except Exception as e:
            return f"Error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
