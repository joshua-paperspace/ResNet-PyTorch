from flask import Flask, request, render_template, redirect, url_for
from resnet34 import resnet34
from preprocess import imgToTensor
import torch
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

MODEL_PATH = './models/cifar_resnet.pth'

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')
    

@app.route('/', methods=['POST'])
def predict_img():

    image = Image.open(request.form['img'])

    filename = 'static/uploads/uploaded_image.png'
    image.save(filename)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet34_test = resnet34(3, 10)
    resnet34_test.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    tensor = imgToTensor(image)
    
    output = resnet34_test(tensor)
    _, predicted = torch.max(output.data, 1)
    prediction = classes[predicted]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')
