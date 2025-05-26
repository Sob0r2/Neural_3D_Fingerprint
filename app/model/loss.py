import torch
import torch.nn.functional as F


def ntxent_loss(anchor_emb, positive_emb, negative_emb, temperature=0.1):
    """
    Computes the NT-Xent contrastive loss for a batch of embeddings.
    """
    positive_sim = F.cosine_similarity(anchor_emb, positive_emb, dim=-1)
    negative_sim = F.cosine_similarity(anchor_emb.unsqueeze(1), negative_emb, dim=-1)

    logits = torch.cat([positive_sim.unsqueeze(1), negative_sim], dim=1)
    logits /= temperature

    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)