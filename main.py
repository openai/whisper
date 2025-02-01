import whisper

model = whisper.load_model('tiny.en')

print(model.transcribe('test_data/5/out001.wav')['text'])
