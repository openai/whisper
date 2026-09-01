import whisper
import time
import datetime
from flask import Flask, request, jsonify
import threading

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 确保中文正常显示

# 全局变量存储模型
model = None


def load_model():
    global model
    print("开始加载 Whisper 模型...")
    start_time = time.time()
    
    # 手动指定模型存储路径
    model_path = "./models"  # 您可以修改为任意路径
    
    # 根据实际情况，选择使用CPU还是GPU
    model = whisper.load_model("medium", download_root=model_path)
    
    load_time = time.time() - start_time
    print(f"模型加载完成，耗时: {str(datetime.timedelta(seconds=load_time))}")
    print(f"模型存储路径: {model_path}")


# 在应用启动时加载模型
@app.before_request
def before_first_request():
    global model
    if model is None:
        print("首次请求，加载模型中...")
        load_model()


@app.route('/transcribe', methods=['POST'])
def transcribe():
    if model is None:
        return jsonify({"error": "模型尚未加载完成"}), 503

    if 'audio' not in request.files:
        return jsonify({"error": "未提供音频文件"}), 400

    audio_file = request.files['audio']
    audio_path = f"/{audio_file.filename}"
    audio_file.save(audio_path)

    # 开始转录 - 使用 FP32 避免 NaN
    start_time = time.time()
    result = model.transcribe(audio_path, language="zh", fp16=False) # 主动降低精度，使用 FP32 避免 NaN
    transcription_time = time.time() - start_time

    return jsonify({
        "text": result["text"],
        "processing_time": transcription_time
    })


@app.route('/transcribe_text', methods=['POST'])
def transcribe_text():
    """返回纯文本格式的转录结果，方便命令行查看"""
    if model is None:
        return "模型尚未加载完成", 503

    if 'audio' not in request.files:
        return "未提供音频文件", 400

    audio_file = request.files['audio']
    audio_path = f"/{audio_file.filename}"
    audio_file.save(audio_path)

    # 开始转录 - 使用 FP32 避免 NaN
    start_time = time.time()
    result = model.transcribe(audio_path, language="zh", fp16=False)
    transcription_time = time.time() - start_time

    # 返回纯文本格式
    return f"{result['text']}\r\n处理时间: {transcription_time:.2f}秒"


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


if __name__ == '__main__':
    # 在启动应用前预先加载模型
    print("启动服务前预先加载模型...")
    load_model()
    app.run(host='0.0.0.0', port=5000, threaded=True)