import torch
import torch.nn as nn
from transformers import AutoModel, ViTModel


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=768, projection_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
        
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected 
        return self.layer_norm(x)
    

class StyleSpaceModel(nn.Module):
    def __init__(self, vision_model_name="google/vit-base-patch16-224", text_model_name="distilbert-base-uncased", projection_dim=512, freeze_backbones=False):
        
        super().__init__()
        
        self.vision_encoder = ViTModel.from_pretrained(vision_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        
        self.vision_projection = ProjectionHead(
            embedding_dim=self.vision_encoder.config.hidden_size, 
            projection_dim=projection_dim
        )
        self.text_projection = ProjectionHead(
            embedding_dim=self.text_encoder.config.hidden_size,     
            projection_dim=projection_dim
        )
        
        if freeze_backbones:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
    def forward(self, images, text_batch):
        
        vision_outputs = self.vision_encoder(pixel_values=images)
        image_features = vision_outputs.last_hidden_state[:, 0, :]
        
        text_outputs = self.text_encoder(
            input_ids = text_batch["input_ids"],
            attention_mask = text_batch["attention_mask"]
        )
        
        text_features = text_outputs.last_hidden_state[:, 0, :]
        
        image_embeddings = self.vision_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        
        image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        
        return image_embeddings, text_embeddings