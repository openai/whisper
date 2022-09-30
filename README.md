# Whisper OpenVINO

This repo is a fork of whisper ASR models with openvino backend. Currently, the transcribe functionality of all models but `large` is supported.

To install, please run the following command with the environment described in the origin repo: https://github.com/openai/whisper.git

```bash
pip install git+https://github.com/zhuzilin/whisper-openvino.git
```

And you can use this modified version of whisper the same as the origin version. For example, to test the performace gain, I transcrible the John Carmack's amazing 92 min talk about rendering at QuakeCon 2013 (you could check the record on [youtube](https://www.youtube.com/watch?v=P6UKhR0T6cs)) with macbook pro 2019 (Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz) with:

```bash
whisper carmack.mp3 --model tiny.en --beam_size 3
```

And the end-to-end time is shown below:

|audio length|origin whisper|whisper openvino|
|-|-|-|
|92 min|67.57 min|39.16 min|

You can check the transcribed txt in [carmack.mp3.txt](./carmack.mp3.txt).

All weights and models include the intermediate ONNX are uploaded to [huggingface model hub](https://huggingface.co/models?search=whisper-openvino).

