from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = [
            float(request.form['Gender']),
            float(request.form['Married']),
            float(request.form['Dependents']),
            float(request.form['Education']),
            float(request.form['Self_Employed']),
            float(request.form['Credit_History']),
            float(request.form['Property_Area']),
            float(request.form['ApplicantIncomelog']),
            float(request.form['LoanAmount_log']),
            float(request.form['Loan_Amount_Term_log']),
            float(request.form['Total_Income_log'])
        ]

        # Make prediction
        prediction = model.predict([np.array(data)])
        result = "Approved" if prediction[0] == 1 else "Rejected"

        return render_template('index.html', prediction=result)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
