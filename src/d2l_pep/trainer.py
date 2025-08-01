import logging
from typing import Callable

import tqdm
from pydantic import BaseModel

from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch


logger = logging.getLogger(__name__)


class _DataLoaderConfig(BaseModel):

    model_config = {
        "extra" : "forbid"
    }

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0


class Trainer:

    def __init__(self, max_epochs: int, num_gpus: int = 0):
        assert num_gpus == 0, "Only CPU training is supported in this version."
        self._max_epochs = max_epochs
        self._dataloader = None

    def fit(self, model: nn.Module, data: Dataset, optimizer: torch.optim.Optimizer, loss_fn: Callable, dataloader_config: dict = {}):

        logger.info("Starting data loading...")
        dataloader_cfg = _DataLoaderConfig(**dataloader_config)

        dataloader = DataLoader(
            data, batch_size=dataloader_cfg.batch_size, shuffle=dataloader_cfg.shuffle, num_workers=dataloader_cfg.num_workers
        )
        logger.info("Data loading complete.")

        pbar = tqdm.tqdm(range(self._max_epochs), ascii=True)
        for _ in pbar:
            running_loss = 0.0
            for batch in dataloader:
                inputs, targets = batch

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
