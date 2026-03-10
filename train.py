import os
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    print(f"Starting training with framework: {cfg.framework}")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from trackshot.models.architecture import build_model
    from trackshot.data.dataset import get_dataloader

    # PyTorch Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    train_loader = get_dataloader(cfg.data, split="train")
    
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(cfg.training.epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

if __name__ == "__main__":
    main()
