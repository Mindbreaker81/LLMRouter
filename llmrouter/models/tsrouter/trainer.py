"""
TSRouterTrainer: Training wrapper for the TSRouter GNN.

Delegates actual training to TSRouterPrediction (which owns the optimizer and
training loop internally).  This class satisfies the LLMRouter BaseTrainer
interface so TSRouter can be used consistently with the rest of the framework.
"""

from __future__ import annotations

from llmrouter.models.base_trainer import BaseTrainer


class TSRouterTrainer(BaseTrainer):
    """
    TSRouterTrainer
    ---------------
    Trainer wrapper for TSRouter.

    TSRouterPrediction (from the TSRouter codebase) manages its own AdamW
    optimizer and training loop internally.  This class simply calls
    ``router.initialize()`` (which triggers graph construction + training)
    and then optionally saves the resulting checkpoint.

    Usage::

        router  = TSRouter(yaml_path)
        trainer = TSRouterTrainer(router, device="cuda")
        trainer.train()   # builds graph, trains GNN, saves checkpoint
    """

    def __init__(self, router, optimizer=None, device: str = "cpu"):
        super().__init__(router=router, optimizer=optimizer, device=device)

    # ------------------------------------------------------------------
    # Required abstract method
    # ------------------------------------------------------------------

    def train(self, dataloader=None):
        """
        Initialize (build graph + train GNN) and save the checkpoint.

        The ``dataloader`` argument is unused; TSRouter consumes its own
        data files as configured via YAML.
        """
        router = self.router

        # Initialize builds the HGT graph and trains from scratch.
        if router._router is None:
            router.initialize(device=self.device)
        else:
            print("[TSRouterTrainer] Router already initialized; skipping re-init.")

        # Persist the trained model.
        router.save_router()
        print("[TSRouterTrainer] Training complete.")
