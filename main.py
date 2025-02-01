import whisper

model = whisper.load_model('tiny.en')

print(model.transcribe('test_data/30s/out001.wav')['text'])
