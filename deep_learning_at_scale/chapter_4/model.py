import evaluate
import torch
from lightning import LightningModule
from transformers import GPT2LMHeadModel  # AutoModelForCausalLM,
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

## Allow TensorCore optimisation
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("medium")


class GPT2Module(LightningModule):
    def __init__(
        self,
        name: str = "gpt2",
        torch_dtype: str = None,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        graph_compile_mode: str = None,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            use_fast=True,
        )
        self.config = AutoConfig.from_pretrained(name)
        self.model = GPT2LMHeadModel.from_pretrained(
            name,
            config=self.config,
            torch_dtype=torch_dtype,
        )
        if graph_compile_mode:
            self.model = torch.compile(self.model, mode=graph_compile_mode)
            torch._dynamo.config.suppress_errors = True

        self.metric = evaluate.load("accuracy")

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        preds = torch.argmax(logits, axis=-1)
        labels = batch["labels"]

        f_labels = labels[:, 1:].reshape(-1)
        f_preds = preds[:, :-1].reshape(-1)

        self.log("val/loss", val_loss, prog_bar=True)
        self.log_dict(
            self.metric.compute(predictions=f_preds, references=f_labels), prog_bar=True
        )

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        test_loss, logits = outputs[:2]

        preds = torch.argmax(logits, axis=-1)
        labels = batch["labels"]

        f_labels = labels[:, 1:].reshape(-1)
        f_preds = preds[:, :-1].reshape(-1)

        self.log("test/loss", test_loss, prog_bar=True)
        self.log("test/perplexity", torch.exp(test_loss))
        self.log_dict(
            self.metric.compute(predictions=f_preds, references=f_labels), prog_bar=True
        )

        return {"loss": test_loss, "preds": preds, "labels": labels}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
