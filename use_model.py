import torch
from torchvision import transforms
from PIL import Image
from model import MyCnn
import os

def predict_image(image_path, model_path='cnn_model_01.pth'):

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyCnn().to(device)

    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Please run main.py first to train and save the model.")
        return

    model.eval()


    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return


    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_idx = torch.max(output, 1)
        predicted_class = classes[predicted_idx.item()]

    print(f"Prediction for '{image_path}': {predicted_class}")


if __name__ == "__main__":
    img_path = './test-data/ho.png'

    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print(f"File not found at: {img_path}")
