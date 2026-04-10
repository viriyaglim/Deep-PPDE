import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List

@dataclass
class BaseOption:
    pass
    
    @abstractmethod
    def payoff(self, x: torch.Tensor, **kwargs):
        ...


class Lookback(BaseOption):
    def __init__(self, lookback_type: str = "put", asset_idx: int = 0):
        """
        lookback_type:
            - 'put'  -> payoff = M_T - S_T
            - 'call' -> payoff = S_T - m_T
        """
        self.lookback_type = lookback_type
        self.asset_idx = asset_idx

    def payoff(self, x):
        """
        x: torch.Tensor of shape (batch_size, N, d)
        returns: torch.Tensor of shape (batch_size, 1)
        """
        s = x[:, :, self.asset_idx]   # single asset path, shape (batch_size, N)
        terminal = s[:, -1]

        if self.lookback_type == "put":
            running_max = torch.max(s, dim=1)[0]
            payoff = running_max - terminal
        elif self.lookback_type == "call":
            running_min = torch.min(s, dim=1)[0]
            payoff = terminal - running_min
        else:
            raise ValueError("lookback_type must be 'put' or 'call'")

        return payoff.unsqueeze(1)

class LookbackPut(BaseOption):
    def __init__(self, asset_idx: int = 0):
        self.asset_idx = asset_idx

    def payoff(self, x):
        s = x[:, :, self.asset_idx]              # (batch_size, N)
        running_max = torch.max(s, dim=1)[0]    # M_T
        terminal = s[:, -1]                     # S_T
        payoff = running_max - terminal         # M_T - S_T
        return payoff.unsqueeze(1)


class LookbackCall(BaseOption):
    def __init__(self, asset_idx: int = 0):
        self.asset_idx = asset_idx

    def payoff(self, x):
        s = x[:, :, self.asset_idx]             # (batch_size, N)
        running_min = torch.min(s, dim=1)[0]    # m_T
        terminal = s[:, -1]                     # S_T
        payoff = terminal - running_min         # S_T - m_T
        return payoff.unsqueeze(1)

class BarrierOption(BaseOption):
    def __init__(self, K, B, option_type="call", barrier_direction="down", knock="out", asset_idx=0):
        self.K = K
        self.B = B
        self.option_type = option_type      # "call" or "put"
        self.barrier_direction = barrier_direction  # "down" or "up"
        self.knock = knock                  # "in" or "out"
        self.asset_idx = asset_idx

    def payoff(self, x):
        """
        x: tensor of shape (batch_size, N, d)
        returns: tensor of shape (batch_size, 1)
        """
        s = x[:, :, self.asset_idx]              # (batch_size, N)
        terminal = s[:, -1]

        if self.option_type == "call":
            vanilla = torch.clamp(terminal - self.K, min=0.0)
        elif self.option_type == "put":
            vanilla = torch.clamp(self.K - terminal, min=0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        running_min = torch.min(s, dim=1)[0]
        running_max = torch.max(s, dim=1)[0]

        if self.barrier_direction == "down":
            hit = (running_min <= self.B).float()
            not_hit = (running_min > self.B).float()
        elif self.barrier_direction == "up":
            hit = (running_max >= self.B).float()
            not_hit = (running_max < self.B).float()
        else:
            raise ValueError("barrier_direction must be 'down' or 'up'")

        if self.knock == "in":
            indicator = hit
        elif self.knock == "out":
            indicator = not_hit
        else:
            raise ValueError("knock must be 'in' or 'out'")

        payoff = vanilla * indicator
        return payoff.unsqueeze(1)
    
class DownAndOutCall(BaseOption):

    def __init__(self, K, B, idx_traded: List[int] = None):
        self.K = K
        self.B = B
        self.idx_traded = idx_traded

    def payoff(self, x):
        """
        x: torch.Tensor of shape (batch_size, N, d)
        returns: torch.Tensor of shape (batch_size, 1)
        """
        if self.idx_traded:
            basket = torch.sum(x[..., self.idx_traded], 2)
        else:
            basket = torch.sum(x, 2)

        terminal = basket[:, -1]
        running_min = torch.min(basket, dim=1)[0]

        vanilla_call = torch.clamp(terminal - self.K, min=0.0)
        alive = (running_min > self.B).float()

        payoff = vanilla_call * alive
        return payoff.unsqueeze(1)
    

class Autocallable(BaseOption):
    
    def __init__(self, idx_traded: int, B: int, Q1: float, Q2: float, q: float, r: float, ts: torch.Tensor):
        """
        Autocallable option with 
        - two observation dates (T/3, 2T/3), 
        - premature payoffs Q1 and Q2
        - redemption payoff q*s
        """
        
        self.idx_traded = idx_traded # index of traded asset
        self.B = B # barrier
        self.Q1 = Q1
        self.Q2 = Q2
        self.q = q # redemption payoff
        self.r = r # risk-free rate
        self.ts = ts # timegrid
    
    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            autocallable option payoff. Tensor of shape (batch_size,1)
        """
        id_t1 = len(self.ts)//3
        mask1 = x[:, id_t1, self.idx_traded]>=self.B
        id_t2 = 2*len(self.ts)//3
        mask2 = x[:, id_t2, self.idx_traded]>=self.B

        payoff = mask1 * self.Q1 * torch.exp(self.r*(self.ts[-1]-self.ts[id_t1])) # we get the payoff Q1, and we put in a risk-less acount for the remaining time
        payoff += ~mask1 * mask2 * self.Q2 * torch.exp(self.r*(self.ts[-1]-self.ts[id_t2]))
        payoff += ~mask1 * (~mask2) * self.q*x[:,-1,self.idx_traded]

        return payoff.unsqueeze(1) # (batch_size, 1)


class EuropeanCall(BaseOption):
    
    def __init__(self, K):
        """
        Parameters
        ----------
        K: float or torch.tensor
            Strike. Id K is a tensor, it needs to have shape (batch_size)
        """
        self.K = K

    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Asset price at terminal time. Tensor of shape (batch_size, d) 
        Returns
        -------
        payoff: torch.Tensor
            basket option payoff. Tensor of shape (batch_size,1)
        """
        if x.dim()==3:
            return torch.clamp(x[:,-1,0]-self.K, 0).unsqueeze(1) # (batch_size, 1)
        elif x.dim() == 2:
            return torch.clamp(x[:,0]-self.K, 0).unsqueeze(1) # (batch_size, 1)
        else:
            raise ValueError('x needs to be last spot price, or trajectory of prices')
