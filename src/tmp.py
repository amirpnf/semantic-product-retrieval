import torch

epoch_checkpoint = torch.load("./checkpoints/stylespace_epoch_5.pt")

torch.save({
    'model_state_dict': epoch_checkpoint['model_state_dict']
}, "./checkpoints/stylespace_best.pt")

print("Fixed best checkpoint structure successfully!")