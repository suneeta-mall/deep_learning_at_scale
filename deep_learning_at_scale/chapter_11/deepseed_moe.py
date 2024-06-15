from pathlib import Path
from typing import List

import deepspeed
import torch
import torch.nn as nn
import torchmetrics
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer

from deep_learning_at_scale.chapter_2.data import MNISTDataModule

__all__: List[str] = []

"""
    deepspeed \
        --num_nodes=1 \
        --num_gpus=2 \
        --bind_cores_to_rank \
        deep_learning_at_scale/chapter_11/mnist_deepspeed.py
"""

ds_config = {
    "train_batch_size": 32,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 0.0002,
        },
    },
    "prescale_gradients": False,
    "wall_clock_breakdown": False,
    "zero_optimization": {
        "stage": 0,
        "allgather_partitions": True,
        "reduce_scatter": True,
        "allgather_bucket_size": 50000000,
        "reduce_bucket_size": 50000000,
        "overlap_comm": True,
        "contiguous_gradients": True,
        "cpu_offload": False,
    },
}


class DeepSpeedMoeWithJitter(nn.Module):
    def __init__(
        self,
        hidden_size=64,
        input_shape=(1, 28, 28),
        num_classes: int = 10,
    ):
        super(DeepSpeedMoeWithJitter, self).__init__()

        channels, width, height = input_shape
        self.block_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        expert = nn.Linear(hidden_size, hidden_size)
        self.moe = deepspeed.moe.layer.MoE(
            hidden_size=hidden_size,
            expert=expert,
            num_experts=6,
            ep_size=2,
            use_residual=False,
            k=2,
            min_capacity=0,
            noisy_gate_policy="Jitter",
        )
        self.post_moe = nn.Sequential(
            nn.Linear(hidden_size, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.block_1(x)
        x, _, _ = self.moe(x)
        x = self.post_moe(x)
        return x


if __name__ == "__main__":
    deepspeed.init_distributed()
    data_dir: Path = Path(".tmp")
    out_dir: Path = Path(".tmp/output")
    max_epochs: int = 40

    exp_name = "chapter_11_ds_moe/train/"
    result_dir = out_dir / exp_name
    result_dir.mkdir(exist_ok=True, parents=True)

    model = DeepSpeedMoeWithJitter()
    datamodule = MNISTDataModule(data_dir, num_workers=4)
    datamodule.setup()

    parameters = split_params_into_different_moe_groups_for_optimizer(
        {
            "params": [p for p in model.parameters() if p.requires_grad],
            "name": "parameters",
        }
    )

    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=None,
        model=model,
        model_parameters=parameters,
        training_data=datamodule.mnist_train,
        config=ds_config,
    )

    local_device = get_accelerator().device_name(model_engine.local_rank)
    local_rank = model_engine.local_rank
    criterion = nn.CrossEntropyLoss()

    for epoch in range(max_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(local_device), data[1].to(local_device)
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            running_loss += loss.item()
            if local_rank == 0 and i % 100 == (99):
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    with torch.no_grad():
        val_loss = []
        val_accuracy = []
        for images, labels in datamodule.val_dataloader():
            labels = labels.to(local_device)
            outputs = model(images.to(local_device))
            _, predicted = torch.max(outputs, 1)
            accuracy = torchmetrics.functional.accuracy(
                predicted, labels, task="multiclass", num_classes=10
            )
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            val_loss.append(loss)
            val_accuracy.append(accuracy)

        print(
            f"""validation accuracy: {torch.stack(val_accuracy).mean()},  
            loss: {torch.stack(val_loss).mean()}"""
        )
