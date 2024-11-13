import torch
from PIL import Image
from torchvision import transforms
from medbasenet.models.medbasenet import MedBaseNet

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = MedBaseNet(num_classes=2)
    model.load_state_dict(torch.load("medbasenet_model.pth"))
    model = model.to(device)
    
    # 预测示例
    image_path = "path/to/test/image.jpg"
    image_tensor = load_image(image_path)
    prediction = predict(model, image_tensor, device)
    
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()
