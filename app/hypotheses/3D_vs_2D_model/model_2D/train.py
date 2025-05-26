import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

import sys
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
sys.path.append(parent_dir)

from model.loss import ntxent_loss

from dotenv import load_dotenv
load_dotenv()

MODELS_PATH = os.getenv("MODELS_PATH")


def train(
    model,
    dataset,
    optimizer,
    device,
    model_name,
    temperature=0.1,
    batch_size=512,
    epochs=100,
):
    """
    Main training loop for the contrastive 2D GNN.
    """
    model.to(device)
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = len(dataset) // batch_size

        with tqdm(total=n_batches, desc=f"Epoch {epoch}", leave=False) as pbar:
            for _ in range(n_batches):
                idx = torch.randint(0, len(dataset), (1,)).item()
                anchor, positive, negatives = dataset[idx]

                anchor = anchor.to(device)
                positive = positive.to(device)
                negatives = [n.to(device) for n in negatives]

                z_anchor = model(anchor)
                z_positive = model(positive)
                z_negative = model(negatives).squeeze(1)
 
                loss = ntxent_loss(z_anchor, z_positive, z_negative, temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": loss.item()})
                pbar.update(1)

        avg_loss = total_loss / n_batches
        losses.append(avg_loss)
       
    # Save model
    Path(MODELS_PATH).mkdir(exist_ok=True)
    model_path = os.path.join(MODELS_PATH, f"{model_name}.pth")
    torch.save(model.state_dict(), model_path)
    return losses
