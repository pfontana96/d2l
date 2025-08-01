import logging

from torch import nn
import torch
from torch.utils.data import Dataset

from d2l_pep.trainer import Trainer
from d2l_pep.log import set_root_logger


logger = logging.getLogger(__name__)


class LinearRegression(nn.Module):

    def __init__(self, input_dim: int, sigma: float = 0.01):

        super().__init__()

        self.w = torch.normal(0, sigma, (input_dim, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b
    

class SyntheticDataset(Dataset):

    def __init__(self, w: torch.Tensor, b: torch.Tensor, num_samples: int = 100, noise_std_dev: float = 0.01):

        self.x = torch.randn(num_samples, w.shape[0])
        self.y = torch.matmul(self.x, w) + b
        self.y_noisy = self.y + noise_std_dev * torch.randn_like(self.y)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y_noisy[idx]


def main():

    set_root_logger(verbose=True)

    # Generate synthetic data
    true_w = torch.tensor([[2.0], [3.0]])
    true_b = torch.tensor([4.0])
    dataset = SyntheticDataset(true_w, true_b, num_samples=1000, noise_std_dev=0.1)

    model = LinearRegression(input_dim=2)
    optimizer = torch.optim.SGD([model.w, model.b], lr=0.01)

    trainer = Trainer(max_epochs=400)
    trainer.fit(model, data=dataset, optimizer=optimizer, loss_fn=nn.MSELoss())

    logger.info(f"Estimated weights: {model.w.data}, Estimated bias: {model.b.data}, True weights: {true_w}, True bias: {true_b}")


if __name__ == "__main__":
    main()