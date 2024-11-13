import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms

class MedicalDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据路径
        self.data = self._load_data()
        
    def _load_data(self):
        data = []
        split = 'train' if self.train else 'test'
        base_path = os.path.join(self.root, split)
        
        for class_id, class_name in enumerate(os.listdir(base_path)):
            class_path = os.path.join(base_path, class_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.png')):
                        img_path = os.path.join(class_path, img_name)
                        data.append((img_path, class_id))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
