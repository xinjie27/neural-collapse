import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from scipy.sparse.linalg import svds


def get_dataset(cfg):
    dataset = cfg["dataset"]
    train = True if cfg["mode"] == "train" else False

    if dataset == "mnist":
        data = datasets.MNIST(root="data", train=train, download=True, transform=ToTensor())
        num_classes = 10
        in_channels = 1
    else:
        # TODO
        data = None
        num_classes = None

    return data, in_channels, num_classes


def train(model, cfg, loader, specs):
    # unpack
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    optimizer = specs.optimizer
    criterion = specs.criterion

    model.train()

    for batch_idx, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # Question: do we want to print out accuracy and loss for each batch? 


    # for epoch in range(num_epochs):
    #     avg_accuracy = 0
    #     avg_loss = 0
    #     for batch_idx, (X, y) in enumerate(train_loader):
    #         X, y = X.to(device), y.to(device)

    #         # Compute prediction error
    #         pred = model(X)
    #         loss_fn = nn.CrossEntropyLoss()
    #         loss = loss_fn(pred, y)

    #         # Backpropagation
    #         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         batch_accuracy = torch.mean((torch.argmax(pred, dim=1) == y).float()).item()
    #         avg_accuracy += batch_accuracy
    #         avg_loss += loss
        
    #     avg_accuracy /= (batch_idx + 1)
    #     avg_loss /= (batch_idx + 1)
    #     print(f"Epoch {epoch + 1}: Loss {avg_loss:.4f}, Accuracy {avg_accuracy:.4f}")

class features:
    pass

def hook(self, input, output):
    features.value = input[0].clone()

def analysis(graphs, model, loader, specs, epoch): 
    # unpack
    num_classes = specs.num_classes
    C = specs.num_classes
    device = specs.device
    criterion_summed = specs.criterion_summed
    weight_decay = specs.weight_decay
    loss_name = specs.loss_name
    classifier = specs.classifier

    model.eval()

    N             = [0 for _ in range(C)]
    mean          = [0 for _ in range(C)]
    Sw            = 0

    loss          = 0
    net_correct   = 0
    NCC_match_net = 0

    for computation in ['Mean','Cov']:
        for batch_idx, (data, target) in enumerate(loader, start=1):

            data, target = data.to(device), target.to(device)

            output = model(data)
            h = features.value.data.view(data.shape[0],-1) # B CHW
            
            # during calculation of class means, calculate loss
            if computation == 'Mean':
                if str(criterion_summed) == 'CrossEntropyLoss()':
                  loss += criterion_summed(output, target).item()
                elif str(criterion_summed) == 'MSELoss()':
                  loss += criterion_summed(output, F.one_hot(target, num_classes=num_classes).float()).item()

            for c in range(C):
                # features belonging to class c
                idxs = (target == c).nonzero(as_tuple=True)[0]
                
                if len(idxs) == 0: # If no class-c in this batch
                  continue

                h_c = h[idxs,:] # B CHW

                if computation == 'Mean':
                    # update class means
                    mean[c] += torch.sum(h_c, dim=0) # CHW
                    N[c] += h_c.shape[0]
                    
                elif computation == 'Cov':
                    # update within-class cov

                    z = h_c - mean[c].unsqueeze(0) # B CHW
                    cov = torch.matmul(z.unsqueeze(-1), # B CHW 1
                                       z.unsqueeze(1))  # B 1 CHW
                    Sw += torch.sum(cov, dim=0)

                    # during calculation of within-class covariance, calculate:
                    # 1) network's accuracy
                    net_pred = torch.argmax(output[idxs,:], dim=1)
                    net_correct += sum(net_pred==target[idxs]).item()

                    # 2) agreement between prediction and nearest class center
                    NCC_scores = torch.stack([torch.norm(h_c[i,:] - M.T,dim=1) \
                                              for i in range(h_c.shape[0])])
                    NCC_pred = torch.argmin(NCC_scores, dim=1)
                    NCC_match_net += sum(NCC_pred==net_pred).item()

        #     pbar.update(1)
        #     pbar.set_description(
        #         'Analysis {}\t'
        #         'Epoch: {} [{}/{} ({:.0f}%)]'.format(
        #             computation,
        #             epoch,
        #             batch_idx,
        #             len(loader),
        #             100. * batch_idx/ len(loader)))
            
        #     if debug and batch_idx > 20:
        #         break
        # pbar.close()
        
        if computation == 'Mean':
            for c in range(C):
                mean[c] /= N[c]
                M = torch.stack(mean).T
            loss /= sum(N)
        elif computation == 'Cov':
            Sw /= sum(N)
    
    graphs.loss.append(loss)
    graphs.accuracy.append(net_correct/sum(N))
    graphs.NCC_mismatch.append(1-NCC_match_net/sum(N))

    # loss with weight decay
    reg_loss = loss
    for param in model.parameters():
        reg_loss += 0.5 * weight_decay * torch.sum(param**2).item()
    graphs.reg_loss.append(reg_loss)

    # global mean
    muG = torch.mean(M, dim=1, keepdim=True) # CHW 1
    
    # between-class covariance
    M_ = M - muG
    Sb = torch.matmul(M_, M_.T) / C

    # avg norm
    W  = classifier.weight
    M_norms = torch.norm(M_,  dim=0)
    W_norms = torch.norm(W.T, dim=0)

    graphs.norm_M_CoV.append((torch.std(M_norms)/torch.mean(M_norms)).item())
    graphs.norm_W_CoV.append((torch.std(W_norms)/torch.mean(W_norms)).item())

    # Decomposition of MSE #
    if loss_name == 'MSELoss':

      wd = 0.5 * weight_decay # "\lambda" in manuscript, so this is halved
      St = Sw+Sb
      size_last_layer = Sb.shape[0]
      eye_P = torch.eye(size_last_layer).to(device)
      eye_C = torch.eye(C).to(device)

      St_inv = torch.inverse(St + (wd/(wd+1))*(muG @ muG.T) + wd*eye_P)

      w_LS = 1 / C * (M.T - 1 / (1 + wd) * muG.T) @ St_inv
      b_LS = (1/C * torch.ones(C).to(device) - w_LS @ muG.T.squeeze(0)) / (1+wd)
      w_LS_ = torch.cat([w_LS, b_LS.unsqueeze(-1)], dim=1)  # c x n
      b  = classifier.bias
      w_ = torch.cat([W, b.unsqueeze(-1)], dim=1)  # c x n

      LNC1 = 0.5 * (torch.trace(w_LS @ (Sw + wd*eye_P) @ w_LS.T) + wd*torch.norm(b_LS)**2)
      LNC23 = 0.5/C * torch.norm(w_LS @ M + b_LS.unsqueeze(1) - eye_C) ** 2

      A1 = torch.cat([St + muG @ muG.T + wd*eye_P, muG], dim=1)
      A2 = torch.cat([muG.T, torch.ones([1,1]).to(device) + wd], dim=1)
      A = torch.cat([A1, A2], dim=0)
      Lperp = 0.5 * torch.trace((w_ - w_LS_) @ A @ (w_ - w_LS_).T)

      MSE_wd_features = loss + 0.5* weight_decay * (torch.norm(W)**2 + torch.norm(b)**2).item()
      MSE_wd_features *= 0.5

      graphs.MSE_wd_features.append(MSE_wd_features)
      graphs.LNC1.append(LNC1.item())
      graphs.LNC23.append(LNC23.item())
      graphs.Lperp.append(Lperp.item())

    # tr{Sw Sb^-1}
    Sw = Sw.cpu().numpy()
    Sb = Sb.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=C-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    graphs.Sw_invSb.append(np.trace(Sw @ inv_Sb))

    # ||W^T - M_||
    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    graphs.W_M_dist.append((torch.norm(normalized_W - normalized_M)**2).item())

    # mutual coherence
    def coherence(V): 
        G = V.T @ V
        G += torch.ones((C,C),device=device) / (C-1)
        G -= torch.diag(torch.diag(G))
        return torch.norm(G,1).item() / (C*(C-1))

    graphs.cos_M.append(coherence(M_/M_norms))
    graphs.cos_W.append(coherence(W.T/W_norms))
    return


def plotting(graphs, specs):
    plt.figure(1)
    plt.semilogy(specs.cur_epochs, graphs.reg_loss)
    plt.legend(['Loss + Weight Decay'])
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss')

    plt.figure(2)
    plt.plot(specs.cur_epochs, 100*(1 - np.array(graphs.accuracy)))
    plt.xlabel('Epoch')
    plt.ylabel('Training Error (%)')
    plt.title('Training Error')

    plt.figure(3)
    plt.semilogy(specs.cur_epochs, graphs.Sw_invSb)
    plt.xlabel('Epoch')
    plt.ylabel('Tr{Sw Sb^-1}')
    plt.title('NC1: Activation Collapse')

    plt.figure(4)
    plt.plot(specs.cur_epochs, graphs.norm_M_CoV)
    plt.plot(specs.cur_epochs, graphs.norm_W_CoV)
    plt.legend(['Class Means','Classifiers'])
    plt.xlabel('Epoch')
    plt.ylabel('Std/Avg of Norms')
    plt.title('NC2: Equinorm')
    
    plt.figure(5)
    plt.plot(specs.cur_epochs, graphs.cos_M)
    plt.plot(specs.cur_epochs, graphs.cos_W)
    plt.legend(['Class Means','Classifiers'])
    plt.xlabel('Epoch')
    plt.ylabel('Avg|Cos + 1/(C-1)|')
    plt.title('NC2: Maximal Equiangularity')

    plt.figure(6)
    plt.plot(specs.cur_epochs,graphs.W_M_dist)
    plt.xlabel('Epoch')
    plt.ylabel('||W^T - H||^2')
    plt.title('NC3: Self Duality')

    plt.figure(7)
    plt.plot(specs.cur_epochs,graphs.NCC_mismatch)
    plt.xlabel('Epoch')
    plt.ylabel('Proportion Mismatch from NCC')
    plt.title('NC4: Convergence to NCC')

    # Plot decomposition of MSE loss
    if specs.loss_name == 'MSELoss':
        plt.figure(8)
        plt.semilogy(specs.cur_epochs, graphs.MSE_wd_features)
        plt.semilogy(specs.cur_epochs, graphs.LNC1)
        plt.semilogy(specs.cur_epochs, graphs.LNC23)
        plt.semilogy(specs.cur_epochs, graphs.Lperp)
        plt.legend(['MSE+wd', 'LNC1', 'LNC2/3', 'Lperp'])
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Decomposition of MSE')

    plt.show()