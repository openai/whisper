# Whisper OpenVINO

This repo is a fork of whisper ASR models with openvino backend. Currently, the transcribe functionality of all models but `large` is supported. The decoder speed should be at least twice as fast as the origin cpu implementation.

To install, please run the following command with the environment described in the origin repo: https://github.com/openai/whisper.git

```
pip install git+https://github.com/zhuzilin/whisper-openvino.git
```

All weights and models are uploaded to huggingface mode hub.

