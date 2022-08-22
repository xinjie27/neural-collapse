import numpy as np
import torch


class NCLoss():
    def __init__(self, cfg):
        self.device = cfg["device"]
        self.num_classes = cfg["num_classes"]
        self.lmbda = cfg["lmbda"] # Regularizing coefficient

    def loss(self, predictions, labels):
        ## General
        predictions = predictions.reshape((predictions.shape[0], -1))
        num_features = predictions.shape[1]
        # Initialize class means and variances
        CMeans = torch.zeros([self.num_classes, num_features]).to(self.device)
        CVars = torch.zeros([self.num_classes, num_features]).to(self.device)

        for c in range(self.num_classes):
            indices = (labels == c).nonzero(as_tuple=True)[0]

            # If no class-c data in this batch, skip
            if len(indices) == 0:
                continue # FIXME

            # Otherwise, compute class means
            predictions_c = predictions[indices]
            CMeans[c] = torch.mean(predictions_c, dim=0) # (num_classes, num_features)
            CVars[c] = torch.mean(torch.square(predictions_c - CMeans[c]), dim=0)
            
        # Compute the global mean
        GMean = torch.mean(CMeans, dim=0) # (num_features,)

        # Compute class norms, each subtracted by GMean first
        CMeans_G = CMeans - GMean
        CNorms = torch.norm(CMeans_G, dim=1)

        ## NC1
        self.nc1_loss = torch.sum(CVars.T / torch.square(CNorms)) # Noise-to-signal TODO: changed CVars to CVars.T
        
        ## NC2
        self.nc2_norm_loss = torch.square(torch.std(CNorms)/torch.mean(CNorms))

        CosMap = torch.zeros([self.num_classes, self.num_classes])
        for c1 in range(self.num_classes):
            for c2 in range(self.num_classes):
                CosMap[c1, c2] = torch.dot(CMeans_G[c1], CMeans_G[c2]) / (CNorms[c1] * CNorms[c2])
        CosMap += (1 / (self.num_classes - 1))
        self.nc2_angle_loss = torch.square(torch.mean(torch.abs(CosMap)))

        # # Compute mutual coherence
        # def _coherence(V): 
        #     G = V.T @ V
        #     G += torch.ones((self.num_classes, self.num_classes),device=self.device) / (self.num_classes - 1)
        #     G -= torch.diag(torch.diag(G))
        #     return torch.norm(G, 1).item() / (self.num_classes * (self.num_classes - 1))

        # self.nc2_angle_loss = _coherence(CMeans.T / CNorms)

        return self.nc1_loss + self.lmbda * (self.nc2_norm_loss + self.nc2_angle_loss)
