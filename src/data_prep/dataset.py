import torch
from PIL import Image
import pandas as pd
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

class StyleSpaceDataset(Dataset):
    
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.image_paths = self.df["image_path"].values
        self.texts = self.df["text"].values
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225]
            )
        ])
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)
        text_string = str(self.texts[idx])
        
        return image_tensor, text_string
        
        
class MultiModalCollate:
    def __init__(self, tokenizer_name="distilbert-base-uncased", max_length=64):
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
    def __call__(self, batch):
        images, texts = zip(*batch)
        images_stacked = torch.stack(images, dim=0)
        
        text_tokenized = self.tokenizer(
            list(texts), 
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return images_stacked, text_tokenized
    

def get_dataloader(csv_path, batch_size=64, num_workers=4):
    
    dataset = StyleSpaceDataset(csv_path)
    collate_fn = MultiModalCollate()
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    import os
    csv_file = "./data/cleaned_dataset.csv"
    if os.path.exists(csv_file):
        print("Testing dataloader pipeline...")
        dl = get_dataloader(csv_file, batch_size=8, num_workers=2)
        for img_batch, text_batch in dl :
            print(f"Image batch tensor shape : {img_batch.shape}")
            print(f"Text input_ids shape : {text_batch['input_ids'].shape}")
            print(f"Text attention_mask shape : {text_batch['attention_mask'].shape}")
            break
    else:
        print(f"Could not find {csv_file}.")