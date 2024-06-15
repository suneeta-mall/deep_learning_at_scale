from typing import List

import timm
import torch
from lightning import LightningModule
from torchmetrics.functional import jaccard_index


class Conv2dReLUWithBN(torch.nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
    ):
        conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,  # Using batch norm so disabled bias to avoid doubling
        )
        relu = torch.nn.ReLU(inplace=True)
        bn = torch.nn.BatchNorm2d(out_channels)

        super(Conv2dReLUWithBN, self).__init__(conv, bn, relu)


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels: int = 0,
    ):
        super().__init__()
        self.conv1 = Conv2dReLUWithBN(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
        )
        self.conv2 = Conv2dReLUWithBN(
            out_channels,
            out_channels,
            kernel_size=3,
        )

    def forward(self, x, skip=None):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                f"""Model depth is {n_blocks}, but you provide `decoder_channels` 
                for {len(decoder_channels)} blocks."""
            )

        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        blocks = [
            DecoderBlock(in_ch, out_ch, skip_ch)
            for in_ch, out_ch, skip_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, *features):
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = head
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class SegmentationModule(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        activation = torch.nn.LogSoftmax(dim=1)
        super().__init__(conv2d, activation)


class UNetSegmentationModel(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        depth: int = 5,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        num_classes: int = 151,
    ):
        super().__init__()
        encoder = timm.create_model(
            "efficientnet_b2",
            pretrained=True,
            features_only=True,
            output_stride=32,
            in_chans=in_channels,
            out_indices=tuple(range(depth)),
        )

        self._encoder_out_channels = encoder.feature_info.channels()
        self.encoder = encoder

        self.decoder = Decoder(
            encoder_channels=self._encoder_out_channels,
            decoder_channels=decoder_channels,
            n_blocks=depth,
        )
        self.segmentation = SegmentationModule(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
        )

    def forward(self, x: torch.Tensor):
        features = self.encoder(x)
        features = self.decoder(*features)
        features = self.segmentation(features)
        return features


class VisionSegmentationModule(LightningModule):
    def __init__(
        self,
        num_classes: int = 151,
        learning_rate: float = 1e-3,
        adam_epsilon: float = 1e-8,
        weight_decay: float = 1e-2,
        use_compile: bool = False,
        use_channel_last: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = UNetSegmentationModel(num_classes=num_classes)

        if use_compile:
            # self.model = torch.compile(self.model,fullgraph=False,mode="max-autotune")
            self.model = torch.compile(self.model)
            # timm/models/efficientnet_blocks.py is not fully compatable for conversion.
            # Allow fallback to eager mode.
            torch._dynamo.config.suppress_errors = True
        if use_channel_last:
            self.model = self.model.to(memory_format=torch.channels_last)

    def forward(self, inputs):
        output = self.model(inputs)
        return output

    def _common_step(self, batch, batch_idx, key: str):
        images, labels = batch
        if self.hparams.use_channel_last:
            images = images.to(memory_format=torch.channels_last)

        outputs = self(images)
        predictions = outputs.argmax(1)

        loss = torch.nn.functional.cross_entropy(outputs, labels.long())
        iou = jaccard_index(
            predictions, labels, task="multiclass", num_classes=self.hparams.num_classes
        )

        # import pdb; pdb.set_trace()
        # import code; code.interact(local=dict(globals(), **locals()))

        self.log(f"{key}/loss", loss, prog_bar=True, sync_dist=key != "train")
        self.log(f"{key}/iou", iou, prog_bar=True, sync_dist=key != "train")

        return {"loss": loss, "preds": predictions, "labels": labels}

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._common_step(batch, batch_idx, "test")

    def configure_optimizers(self):
        # return bnb.optim.Adam8bit(model.parameters(), lr=0.001, betas=(0.9, 0.995))
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
            weight_decay=self.hparams.weight_decay,
        )
        return [optimizer]
