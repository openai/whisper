import sys
import time
import torch
import whisper
from whisper.transcribe import transcribe

#torch.jit.set_fusion_strategy([("DYNAMIC", 1)])
#model = whisper.load_model("large")
model = whisper.load_model("medium.en")
model.eval()

a = time.time()
with torch.no_grad():
    result = transcribe(model, sys.argv[1], verbose=False)
    print(result["text"])
print(time.time() - a)
