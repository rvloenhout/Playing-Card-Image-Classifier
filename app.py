from flask import Flask, render_template, request
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from model import CardClassifier

# Create Flask app
app = Flask(__name__)

# Load the trained model
model = CardClassifier()
model.load_state_dict(torch.load("../../Model/card_classifier.pth", map_location=torch.device('cpu')))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define the classes
image_folder_dataset = ImageFolder("../../playing_cards/train")
classes = [image_folder_dataset.classes[i] for i in range(len(image_folder_dataset.classes))]

# Define the index route
@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        img = Image.open(file.stream).convert("RGB")
        img = transform(img)
        img = img.unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(img)
            _, predicted_class_idx = torch.max(output, 1)

        predicted_class = classes[predicted_class_idx.item()]

    return render_template('index.html', predicted_class=predicted_class)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=4996)
