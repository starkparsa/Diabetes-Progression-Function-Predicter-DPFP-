from flask import Flask, request, render_template
from DPFP import XGB

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', DPF=None, DPF_percentage=None, error=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from the form
        Preg = float(request.form['Pregnancies'])
        Glu = float(request.form['Glucose'])
        BP = float(request.form['BloodPressure'])
        ST = float(request.form['SkinThickness'])
        Insu = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        Age = float(request.form['Age'])

        # Get prediction
        DPF, DPF_percentage = XGB(Preg, Glu, BP, ST, Insu, BMI, Age)
        
        # Render the template with the prediction results
        return render_template('index.html', DPF=DPF, DPF_percentage=DPF_percentage, error=None)
    
    except Exception as e:
        return render_template('index.html', DPF=None, DPF_percentage=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
