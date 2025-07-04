from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model using a path relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'insurance_model.pkl')
model = joblib.load(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    # Default values
    form_data = {
        'age': 30,
        'sex': 'female',
        'bmi': 25.0,
        'children': 0,
        'smoker': 'no',
        'region': 'northeast'
    }
    if request.method == 'POST':
        form_data['age'] = int(request.form['age'])
        form_data['sex'] = request.form['sex']
        form_data['bmi'] = float(request.form['bmi'])
        form_data['children'] = int(request.form['children'])
        form_data['smoker'] = request.form['smoker']
        form_data['region'] = request.form['region']
        sex = 1 if form_data['sex'] == 'male' else 0
        smoker = 1 if form_data['smoker'] == 'yes' else 0
        input_dict = {
            'age': form_data['age'],
            'sex': sex,
            'bmi': form_data['bmi'],
            'children': form_data['children'],
            'smoker': smoker,
            'region_northeast': 1 if form_data['region'] == 'northeast' else 0,
            'region_northwest': 1 if form_data['region'] == 'northwest' else 0,
            'region_southeast': 1 if form_data['region'] == 'southeast' else 0,
            'region_southwest': 1 if form_data['region'] == 'southwest' else 0,
        }
        input_df = pd.DataFrame([input_dict])
        pred = model.predict(input_df)[0]
        prediction = f"${pred:,.2f}"
    return render_template('index.html', prediction=prediction, form_data=form_data)

if __name__ == '__main__':
    app.run(debug=True)
