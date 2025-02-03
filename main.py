import whisper
import os

TEST_DATA_BASE = "test_data/"
FIVE_SEC_BASE = os.path.join(TEST_DATA_BASE, "5s/")
THIRTY_SEC_BASE = os.path.join(TEST_DATA_BASE, "30s/")
TRANSCRIPTS_BASE = "test_transcripts_before/"

model = whisper.load_model("tiny.en")

def transcribe_baseline(file_name):
  return model.transcribe(file_name).get('text', '')

def get_all_files(base_path, count=1000):
  return [os.path.join(base_path, f"out{i:03d}.wav") for i in range(count)]

def write_to_file(file_name, text):
  os.makedirs(os.path.dirname(file_name), exist_ok=True)
  with open(file_name, "w") as f:
    f.write(text)

def calculate_wer(hypothesis, actual):
  hyp_words = hypothesis.strip().lower().split()
  act_words = actual.strip().lower().split()

  dp = [[0] * (len(hyp_words) + 1) for _ in range(len(act_words) + 1)]

  for i in range(len(act_words) + 1):
    dp[i][0] = i
  for j in range(len(hyp_words) + 1):
    dp[0][j] = j

  for i in range(1, len(act_words) + 1):
    for j in range(1, len(hyp_words) + 1):
      if act_words[i - 1] == hyp_words[j - 1]:
        dp[i][j] = dp[i - 1][j - 1]
      else:
        dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)

  total_words = len(act_words)
  return dp[len(act_words)][len(hyp_words)] / total_words if total_words else float("inf") if hyp_words else 0.0

def process_files(files, output_base):
  for file_name in files:
    hypothesis = transcribe_baseline(file_name)
    sample_name = os.path.splitext(os.path.basename(file_name))[0]
    write_to_file(os.path.join(output_base, f"{sample_name}.txt"), hypothesis)

if __name__ == "__main__":
  # process_files(get_all_files(FIVE_SEC_BASE), os.path.join(TRANSCRIPTS_BASE, "5s"))
  process_files(get_all_files(THIRTY_SEC_BASE), os.path.join(TRANSCRIPTS_BASE, "30s"))
