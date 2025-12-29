# far/data/custom_adapter.py
from typing import Any, Dict
import torch
from torch.utils.data import Dataset

class CustomSequenceAdapter(Dataset):
    """
    Wrap a base dataset that yields (context, target) sequences.
    Assumptions (by request):
      - context: (N, C, H, W) float in [0,1]
      - target : (M, C, H, W) float in [0,1]
      - sizes are already correct; no transforms here.

    Returns dict:
      {"video": (T, C, H, W), "path": str(idx)}
    where T = N + M
    """
    def __init__(self, base_dataset: Dataset, context_frames: int, predict_frames: int):
        self.base = base_dataset
        self.N = int(context_frames)
        self.M = int(predict_frames)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.base[idx]
        ts = self.base.timestamps[idx]
        if isinstance(item, dict):
            ctx, tgt = item["context"], item["target"]
        else:
            ctx, tgt = item  # expect a 2-tuple
        if isinstance(ctx, tuple):
            ctx_vid, ctx_irr = ctx[0], ctx[1]
        else:
            ctx_vid, ctx_irr = ctx, torch.empty((ctx.shape[0],), dtype=torch.float32)
        if isinstance(tgt, tuple):
            tgt_vid, tgt_irr = tgt[0], tgt[1]
        else:
            tgt_vid, tgt_irr = tgt, torch.empty((ctx.shape[0],), dtype=torch.float32)
        video = torch.cat([ctx_vid, tgt_vid], dim=0)  # (T, C, H, W)
        irr = torch.cat([ctx_irr, tgt_irr], dim=0)
        
        return {
            "video": video,
            "irradiance": irr,
            "path": str(idx),
            "index": torch.tensor(ts.value, dtype=torch.long),
        }
