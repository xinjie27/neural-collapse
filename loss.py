import numpy as np
import torch


class NCLoss():
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.num_classes = cfg["num_classes"]
        self.batch_size = cfg["batch_size"]
        self.lmbda = cfg["lmbda"] # Regularizing coefficient

    def loss(self, predictions, labels):
        ## General
        predictions = predictions.reshape(self.batch_size, -1)
        num_features = predictions.shape[1]
        # Initialize class means
        CMeans = torch.zeros([self.num_classes, num_features]).to(self.device)

        for c in range(self.num_classes):
            indices = (labels == c).nonzero(as_tuple=True)[0]

            # If no class-c data in this batch, skip
            if len(indices) == 0:
                continue # FIXME

            # Otherwise, compute class means
            predictions_c = predictions[indices]
            CMeans[c] = torch.mean(predictions_c, dim=0) # (num_classes, num_features)

            # Compute the global mean
            GMean = torch.mean(CMeans, dim=0) # (num_features,)

        ## NC1
        

        ## NC2 (Act as a regularizer)
        # Compute class norms, each subtracted by GMean first
        CNorms = torch.norm(CMeans - GMean, dim=1)
        self.nc2_norm_loss = torch.std(CNorms)/torch.mean(CNorms)

        # Compute mutual coherence
        def _coherence(V): 
            G = V.T @ V
            G += torch.ones((self.num_classes, self.num_classes),device=self.device) / (self.num_classes - 1)
            G -= torch.diag(torch.diag(G))
            return torch.norm(G, 1).item() / (self.num_classes * (self.num_classes - 1))
        self.nc2_angle_loss = _coherence(CMeans.T / CNorms)
