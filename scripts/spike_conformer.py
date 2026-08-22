"""Day-one spike: does IndicConformer expose both CTC and RNNT decoder heads?"""

import numpy as np
import torch
from transformers import AutoModel

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"

model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
wav = torch.from_numpy(np.zeros(16000, dtype=np.float32)).unsqueeze(0)

for decoding in ("ctc", "rnnt"):
    try:
        out = model(wav, "hi", decoding)
        print(f"{decoding}: OK -> {out!r}")
    except Exception as exc:
        print(f"{decoding}: UNAVAILABLE -> {type(exc).__name__}: {exc}")
