from flask import Flask, request, render_template, redirect, url_for
from resnet import resnet18
from preprocess import imgToTensor
import torch
from PIL import Image

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

MODEL_PATH = './models/resnet18-epochs-5.pth'

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
    model = resnet18(3, 10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    tensor = imgToTensor(image)
    
    output = model(tensor)
    _, predicted = torch.max(output.data, 1)
    print(predicted)
    prediction = classes[predicted]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='8000')
