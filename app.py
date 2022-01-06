from flask import Flask,request,render_template,redirect
from io import BytesIO
from PIL import Image

from utils import process_pil

import torch

from model import Network

import matplotlib.pyplot as plt

app = Flask(__name__)

net = Network()

net.load_()

@app.route('/',methods=['GET','POST'])
def index():
    if request.method=='POST':

        original = request.files['original']
        fake = request.files['fake']

        if original.filename == '' or fake.filename == '':
            return redirect('/')

        original = Image.open(BytesIO(original.stream.read()))
        fake = Image.open(BytesIO(fake.stream.read()))

        original = process_pil(original)
        fake = process_pil(fake)

        pred = torch.softmax(net(original,fake).squeeze(),0) #get probabilities
        out = pred.argmax()

        return f'{"Forged" if out==1 else "Same"}. confidence: {pred.max()*100 :.2f}%'

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)