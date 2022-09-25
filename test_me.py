import whisper
import time 
import torchdynamo
from whisper.decoding import DecodingTask
paths = []
for i in range(1, 9):
    path = f"david-copperfield-002-chapter-1-i-am-born.2935_silence_0{i}.mp3"
    paths.append(path)

model = whisper.load_model("base", dynamo=torchdynamo.optimize("inductor"))

audios = []
mels = []
for path in paths:
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
    audios.append(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)
    mels.append(mel)

options = whisper.DecodingOptions()

task = DecodingTask(model, options)

# Preheat, so inductor can compile
task.run(mels[0])

start = time.time()
for i in range(1, len(mels)):
    mel = mels[i]
    result = task.run(mel)
    print(result[0].text)

end = time.time()
print(end - start, "seconds")
