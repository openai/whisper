import whisper
import time
import datetime


def format_time(seconds):
    """将秒数格式化为易读的时间字符串"""
    return str(datetime.timedelta(seconds=seconds))


def transcribe_with_timing():
    # 记录开始时间
    start_time = time.time()

    print("开始加载 Whisper 模型...")
    model_load_start = time.time()
    model = whisper.load_model("medium") #
    model_load_time = time.time() - model_load_start
    print(f"模型加载完成，耗时: {format_time(model_load_time)}")

    print("开始语音识别...")
    transcription_start = time.time()
    result = model.transcribe("dingzhen.wav", language="zh")
    transcription_time = time.time() - transcription_start
    print(f"语音识别完成，耗时: {format_time(transcription_time)}")

    # 输出结果
    print("\n识别结果:")
    print(result["text"])

    # 计算总时间
    total_time = time.time() - start_time
    print(f"\n总运行时间: {format_time(total_time)}")
    print(f"详细时间:")
    print(f"- 模型加载: {format_time(model_load_time)} ({model_load_time / total_time:.1%})")
    print(f"- 语音识别: {format_time(transcription_time)} ({transcription_time / total_time:.1%})")


if __name__ == "__main__":
    transcribe_with_timing()