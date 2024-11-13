import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from medbasenet.models.medbasenet import MedBaseNet
from medbasenet.utils.data_loader import MedicalDataset

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = MedBaseNet(num_classes=2).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 加载数据
    train_dataset = MedicalDataset(root="path/to/data", train=True)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}")
        
    # 保存模型
    torch.save(model.state_dict(), "medbasenet_model.pth")

if __name__ == "__main__":
    main()
