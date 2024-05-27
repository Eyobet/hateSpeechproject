from flask import Flask, render_template
from flask import request
import pickle

tokenizer = pickle.load(open('models/cv.pkl', 'rb'))
model = pickle.load(open('models/clf.pkl', 'rb'))
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        text = request.form.get('note')
        return render_template('home.html', text=text)
    else:
        return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form.get('note')
        tokenized_note = tokenizer.transform([text])  # Note: Wrap text in a list
        predictions = model.predict(tokenized_note)
        predictions = 1 if predictions == 1 else -1
        return render_template('home.html', text=text, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
