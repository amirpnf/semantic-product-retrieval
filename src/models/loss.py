import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / init_temperature)))
        
    def forward(self, image_embeddings, text_embeddings):
        device = image_embeddings.device
        batch_size = image_embeddings.size(0)
        
        scale = torch.clamp(self.logit_scale.exp(), max=100.0)
        
        logits_per_image = torch.matmul(image_embeddings, text_embeddings.T)
        
        logits_per_image = logits_per_image * scale
        logits_per_text = logits_per_image.T
        
        targets = torch.arange(batch_size, dtype=torch.long, device=device)
        
        loss_images = F.cross_entropy(logits_per_image, targets)
        loss_texts = F.cross_entropy(logits_per_text, targets)
        
        return (loss_images + loss_texts) / 2