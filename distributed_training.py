"""
Heavily inspired by
https://towardsdatascience.com/writing-distributed-applications-with-pytorch-f5de4567ed3b
"""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from pelutils import TickTock, log, Levels
from tqdm import tqdm


TT = TickTock()
PORT = "3090"
N = 10 ** 4
n = 10 ** 3
inp = torch.randn(N, n, 10)
labs = torch.randn(N, n, 5)


def setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = PORT

    if rank != -1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup(rank):
    if rank != -1:
        dist.destroy_process_group()

def _is_master(rank: int):
    return rank < 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(10, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with TT.profile("Forward"):
            return self.l2(self.relu(self.l1(x)))

def demo_basic(rank: int, world_size: int):
    log.configure(
        f"multigpu{rank}.log",
        f"Rank {rank} des multi-GPUs",
        print_level=Levels.DEBUG if _is_master(rank) else None,
    )
    setup(rank, world_size)
    # Device and model
    if rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model().to(device)
    else:
        device = torch.device("cuda", index=rank)
        model = DDP(Model().to(device), device_ids=[rank])
    model.train()

    # Data
    train_dataset = TensorDataset(inp, labs)
    if rank != -1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size)
    else:
        train_sampler = None
    train_dataloader = DataLoader(train_dataset, batch_size=N//100, sampler=train_sampler)

    # Optimizer and loss
    lf = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=0.001)
    optim.zero_grad()

    # Twain
    for epoch in range(100):
        total_loss = 0
        TT.tick()
        for i, (batch_inp, batch_labs) in enumerate(train_dataloader):
            batch_inp, batch_labs = batch_inp.to(device), batch_labs.to(device)
            optim.zero_grad()
            out = model(batch_inp)
            loss = lf(out, batch_labs)
            total_loss += loss.item()
            loss.backward()
            optim.step()
        log(
            f"Rank = {rank}, epoch = {epoch}",
            f"Avg. loss = {total_loss/(i+1)}",
            f"Training time = {TT.stringify_time(TT.tock())}",
        )

    cleanup(rank)

def run_demo(demo_fn, world_size: int):
    mp.spawn(
        demo_fn,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )

if __name__ == "__main__":
    with log.log_errors:
        if torch.cuda.device_count() > 1:
            run_demo(demo_basic, torch.cuda.device_count())
        else:
            demo_basic(-1, 1)
