import typer

from .deepspeed_zero_3 import app as zero3
from .rpc_torch_exp_pipeline_ddp_deepfm import app as rpc_pt_pipeline_hybrid_deepfm
from .rpc_torch_exp_pipeline_deepfm import app as rpc_torch_pipeline_deepfm_exp
from .rpc_torch_pipeline_ddp_deepfm import app as rpc_pt_hybrid_deepfm
from .rpc_torch_pipeline_deepfm import app as rpc_pt_pipeline_deepfm
from .torch_baseline_deepfm import app as pt_baseline_deepfm
from .torch_model_parallel_deepfm import app as pt_mp_deepfm
from .torch_pipeline_deepfm import app as pt_pipe_deepfm
from .torch_tensor_fsdp_deepfm import app as pt_tensor_ddp_deepfm
from .torch_tensor_deepfm import app as pt_tensor_deepfm

__all__ = ["app"]

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)

app.add_typer(pt_baseline_deepfm, name="pt-baseline-deepfm")
app.add_typer(pt_mp_deepfm, name="pt-mp-deepfm")
app.add_typer(pt_pipe_deepfm, name="pt-pipe-deepfm")
app.add_typer(pt_tensor_deepfm, name="pt-tensor-deepfm")
app.add_typer(pt_tensor_ddp_deepfm, name="pt-tensor-ddp-deepfm")

app.add_typer(rpc_pt_pipeline_deepfm, name="rpc-pt-pipeline-deepfm")
app.add_typer(rpc_pt_hybrid_deepfm, name="rpc-pt-hybrid-deepfm")

app.add_typer(rpc_torch_pipeline_deepfm_exp, name="rpc-pt-exp-pipeline-deepfm")
app.add_typer(rpc_pt_pipeline_hybrid_deepfm, name="rpc-pt-exp-pipeline-hybrid-deepfm")

app.add_typer(zero3, name="zero3")


@app.command()
def init():
    print("echo")
    return 0
