import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class SequenceMixer(nn.Module, ABC):
    """
    Interface abstraite pour tout composant qui mélange l'information temporelle.
    Cela peut être une Attention, un SSM (Mamba), une Convolution, etc.
    """
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[dict] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor (Batch, Seq_Len, Dim)
            mask: Masque d'attention ou de padding
            inference_params: Cache pour l'inférence autoregressive (KV-cache ou SSM state)
        """
        pass

class PositionalEncoding(nn.Module, ABC):
    """Interface pour l'injection de position."""
    @abstractmethod
    def forward(self, x: torch.Tensor, step_offset: int = 0) -> torch.Tensor:
        pass
