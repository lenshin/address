#!python3
import json

from flask import Flask, request, render_template
from address_parser import AddressParser

app = Flask(__name__)
model = AddressParser()

@app.route('/', methods=['GET', 'POST'])
def predict():
    address = ''
    prediction = ''
    if request.method == 'POST' and 'address' in request.form:
        address = request.form.get('address')
        prediction = model(address)
        prediction = json.dumps(prediction, indent=2, ensure_ascii=False)
    return render_template('index.html', address=address, prediction=prediction)
app.run()