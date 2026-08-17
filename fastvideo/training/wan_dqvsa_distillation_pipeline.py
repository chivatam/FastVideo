# SPDX-License-Identifier: Apache-2.0
"""DQ-VSA Stage-2 recovery: velocity distillation from a frozen P4G-operator teacher.

Distilled Quantization-Aware VSA (see
``artifacts/sparsefp4_native/TRAINING_RECOVERY_PLAN.md``). Teacher and student
share weights at init and the identical VSA256 mask policy; they differ ONLY
in fine-branch QK precision:

  teacher: frozen BF16 sparse model (P4G operator; ``fine_qat=False``)
  student: same operator with production fake-quant NVFP4 QK + STE
           (``SPARSEFP4_QAT_VSA256_ATTN`` backend, ``fine_qat=True``)

Loss (velocity distillation, teacher/student see identical x_t, t, text):

  L_QVD = || u_student_sparse_nvfp4(x_t,t,c) - u_teacher_sparse_bf16(x_t,t,c) ||^2

The ground-truth flow-matching loss is logged for diagnostics but carries no
gradient. Launch like ``wan_training_pipeline`` with
``FASTVIDEO_ATTENTION_BACKEND=SPARSEFP4_QAT_VSA256_ATTN``.
"""
import os
import sys

import torch
import torch.distributed as dist

from fastvideo import envs
from fastvideo.distributed import get_world_group
from fastvideo.fastvideo_args import FastVideoArgs, TrainingArgs
from fastvideo.forward_context import set_forward_context
from fastvideo.logger import init_logger
from fastvideo.pipelines.pipeline_batch_info import TrainingBatch
from fastvideo.training.wan_training_pipeline import WanTrainingPipeline
from fastvideo.utils import maybe_download_model, verify_model_config_and_directory

logger = init_logger(__name__)


class WanDQVSADistillationPipeline(WanTrainingPipeline):
    """Wan training pipeline with a frozen sparse-BF16 teacher velocity target."""

    def initialize_training_pipeline(self, training_args: TrainingArgs) -> None:
        assert envs.FASTVIDEO_ATTENTION_BACKEND == "SPARSEFP4_QAT_VSA256_ATTN", (
            "DQ-VSA distillation requires FASTVIDEO_ATTENTION_BACKEND="
            "SPARSEFP4_QAT_VSA256_ATTN")
        super().initialize_training_pipeline(training_args)

        from fastvideo.attention.backends.sparsefp4_qat_vsa256 import set_fine_qat
        n_student = set_fine_qat(self.transformer, True)

        # Teacher: fresh copy of the pretrained weights, frozen, BF16 fine QK.
        self.teacher_transformer = self._load_teacher(training_args)
        self.teacher_transformer.requires_grad_(False)
        self.teacher_transformer.eval()
        n_teacher = set_fine_qat(self.teacher_transformer, False)
        if n_student == 0 or n_teacher == 0:
            raise RuntimeError(f"DQ-VSA impl discovery failed (student={n_student}, teacher={n_teacher}); "
                               "both models must build with SPARSEFP4_QAT_VSA256_ATTN")
        logger.info("DQ-VSA: student fake-quant impls=%d, teacher BF16 impls=%d", n_student, n_teacher)

        self._grad_check = os.environ.get("FASTVIDEO_DQVSA_GRAD_CHECK", "0") == "1"

    def _load_teacher(self, training_args: TrainingArgs) -> torch.nn.Module:
        """Load a second transformer instance from the pretrained checkpoint."""
        from fastvideo.models.loader.component_loader import PipelineComponentLoader

        model_path = maybe_download_model(training_args.model_path)
        config = verify_model_config_and_directory(model_path)
        if "transformer" not in config:
            raise ValueError(f"transformer not found in model config at {model_path}")
        transformers_or_diffusers, _ = config["transformer"]
        # NOTE: deliberately NOT setting ``_loading_teacher_critic_model`` here.
        # That DMD flag forces the teacher onto automatic (dense) attention
        # selection; the DQ-VSA teacher must build with the same
        # SPARSEFP4_QAT_VSA256_ATTN operator as the student (fine_qat is
        # flipped to False after loading).
        teacher = PipelineComponentLoader.load_module(
            module_name="transformer",
            component_model_path=os.path.join(model_path, "transformer"),
            transformers_or_diffusers=transformers_or_diffusers,
            fastvideo_args=training_args,
        )
        return teacher

    def _transformer_forward_and_compute_loss(self, training_batch: TrainingBatch) -> TrainingBatch:
        assert training_batch.attn_metadata is not None, ("DQ-VSA requires VSA256 attention metadata")
        input_kwargs = training_batch.input_kwargs
        assert not self.train_transformer_2, "DQ-VSA supports the single-transformer Wan models"

        with self.tracker.timed("timing/forward_backward"), set_forward_context(
                current_timestep=training_batch.current_timestep, attn_metadata=training_batch.attn_metadata):
            with torch.no_grad():
                teacher_pred = self.teacher_transformer(**input_kwargs)

            model_pred = self.transformer(**input_kwargs)
            assert model_pred.shape == teacher_pred.shape

            loss = (torch.mean(
                (model_pred.float() - teacher_pred.float())**2) / self.training_args.gradient_accumulation_steps)
            loss.backward()
            avg_loss = loss.detach().clone()

            # Diagnostics only: distance of student velocity to the ground truth.
            with torch.no_grad():
                assert training_batch.latents is not None and training_batch.noise is not None
                gt_target = training_batch.noise - training_batch.latents
                gt_loss = torch.mean((model_pred.float() - gt_target.float())**2)

        if self._grad_check:
            bad = [
                name for name, p in self.transformer.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            if bad:
                raise RuntimeError(f"non-finite gradients in: {bad[:10]}")
            logger.info("DQ-VSA grad check: all gradients finite "
                        "(distill_loss=%.6f, gt_flow_loss=%.6f)", avg_loss.item(), gt_loss.item())

        with self.tracker.timed("timing/reduce_loss"):
            world_group = get_world_group()
            avg_loss = world_group.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
        training_batch.total_loss += avg_loss

        return training_batch


def main(args) -> None:
    logger.info("Starting DQ-VSA velocity-distillation pipeline...")
    pipeline = WanDQVSADistillationPipeline.from_pretrained(args.pretrained_model_name_or_path, args=args)
    pipeline.train()
    logger.info("DQ-VSA pipeline done")


if __name__ == "__main__":
    argv = sys.argv  # noqa: F841
    from fastvideo.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = FastVideoArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
