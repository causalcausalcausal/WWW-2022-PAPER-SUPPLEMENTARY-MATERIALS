import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class DGB(nn.Module):
    """Get the total values from Dual-Gradient-Bisection(DGB).

    Parameters
        budget : int
        cost : ndarray
        value : ndarray
        
    Returns:
        sum_values : float
            total values from DGB
        sum_cost : float
            total cost from DGB

    """
    def __init__(self, budget, cost, value):
        super(DGB, self).__init__()
        self.budget = budget
        self.treats = cost
        self.uplifts = value 
        self.lam = nn.Parameter(torch.tensor(1.1), requires_grad=True)
        
    def sub_dualLR(self):
        n, k = self.uplifts.shape
        treats = torch.tensor(self.treats)
        values = torch.tensor(self.uplifts)
        tmp = values - self.lam * treats
        y_lam = torch.relu(tmp.max(dim=1).values).sum()        
        return y_lam    

    def calculate_loss(self, loss_print=False):
        tmp = self.budget * self.lam + self.sub_dualLR()
        n, _ = self.uplifts.shape
        return tmp/n
    
    # when cost is same for each user
    def generate_decisions(self):
        n, k = self.uplifts.shape
        lam = self.lam.detach().numpy()
        tmp = self.uplifts - lam * self.treats
        cols = tmp.argmax(axis=1)
        indices, values, spend = [], [], []
        for i, j in zip(range(n), cols):
            if tmp[i, j] > 0:
                indices.append((i, j))
                values.append(self.uplifts[i, j])
                spend.append(self.treats[j])
        sum_values = sum(values)
        sum_cost = sum(spend)
        return indices, values, spend 
    
    def save_to_dataframe(self,df):
#         df = pd.DataFrame()
        a, b, c = self.generate_decisions()
        vals = [0.0 for i in range(len(self.uplifts))]
        for ind, val in zip(a, b):
            i, j = ind
            vals[i] = vals[i] + val
        df['values'] = vals   
        # the cost of control group is 0.1
        spends = [0.1 for i in range(len(self.uplifts))]
        for ind, spend in zip(a, c):
            i, j = ind
            spends[i] = spends[i] + spend
        df['cost'] = spends
        return df

    def train(self):
        epoch = 100
        iters = 1
        eps = 0.0001
        losses = []
        grads = []
        self.roi = self.uplifts/self.treats
        self.lam_max =self.roi.max()
        lam1, lam2 = torch.tensor(0.0), torch.tensor(self.lam_max)

        for e in range(epoch):
            if (lam1-lam2).abs() < eps:
                break
            for i in range(iters):
                lam = (lam1 + lam2)/2
                self.lam.data = lam
                loss = self.calculate_loss(loss_print=False)
                losses.append(loss)
                loss.backward()
                if self.lam.grad > 0:
                    lam2 = lam
                else:
                    lam1 = lam
                grads.append(self.lam.grad.item())
                self.lam.grad.zero_()