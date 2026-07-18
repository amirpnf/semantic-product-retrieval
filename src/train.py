import os
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataset import get_dataloader
from model import StyleSpaceModel
from loss import ContrastiveLoss

CSV_PATH = "./data/cleaned_dataset.csv"
BATCH_SIZE = 16          
EPOCHS = 5
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-2
PROJECTION_DIM = 512
CHECKPOINT_DIR = "./checkpoints"

def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device):
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for step, (images, texts) in enumerate(pbar):
        images = images.to(device)
        texts = {k: v.to(device) for k, v in texts.items()}
        
        optimizer.zero_grad(set_to_none=True) 
        
        with torch.cuda.amp.autocast():
            img_embeds, txt_embeds = model(images, texts)
            loss = loss_fn(img_embeds, txt_embeds)
            
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}", 
            "scale": f"{loss_fn.logit_scale.exp().item():.2f}", 
            "lr": f"{current_lr:.2e}"
        })
        
    return total_loss / len(dataloader)

def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using training device: {device}")
    if device.type != 'cuda':
        print("WARNING: CUDA is not available")
        
    print("Loading datasets...")
    train_loader = get_dataloader(CSV_PATH, batch_size=BATCH_SIZE, num_workers=4)
    
    print("Initializing models...")
    model = StyleSpaceModel(projection_dim=PROJECTION_DIM).to(device)
    loss_fn = ContrastiveLoss().to(device)
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [loss_fn.logit_scale],
            "weight_decay": 0.0,
            "lr": LEARNING_RATE
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_loader), eta_min=1e-6)
    
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"Starting training for {EPOCHS} epochs...\n")
    best_loss = float("inf")
    
    for epoch in range(1, EPOCHS + 1):
        print(f"--- Epoch {epoch}/{EPOCHS} ---")
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        scheduler.step()
        
        print(f"Epoch {epoch} finished. Average Loss: {epoch_loss:.4f}\n")
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"stylespace_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(CHECKPOINT_DIR, "stylespace_best.pt")
            torch.save({
                'model_state_dict': model.state_dict()
            }, best_path)
            print(f"New best checkpoint saved to {best_path}")
            
    print("Training complete. Ready for vector indexing.")

if __name__ == "__main__":
    main()