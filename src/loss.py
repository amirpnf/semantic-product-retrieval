import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        
    def forward(self, image_embeddings, text_embeddings):
        
        device = image_embeddings.device
        batch_size = image_embeddings.size(0)
        
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.T)
        
        t = torch.clamp(self.temperature, min=0.001, max=0.5)
        logits_per_image = logits_per_image / t
        logits_per_text = logits_per_image.T
        
        targets = torch.arange(batch_size, dtype=torch.long, device=device)
        
        loss_images = F.cross_entropy(logits_per_image, targets)
        loss_texts = F.cross_entropy(logits_per_text, targets)
        
        return (loss_images + loss_texts) / 2