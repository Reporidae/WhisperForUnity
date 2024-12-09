import os
import sys
import re
import socket
import torch
import json
import warnings
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.signal import resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from transformers import logging
import time

# 특정 경고 메시지 무시
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Transformers 로그 수준 설정
logging.set_verbosity_error()

# PyInstaller 임시 디렉터리 경로 처리
if getattr(sys, 'frozen', False):  # PyInstaller로 빌드된 경우
    current_dir = os.path.dirname(sys.executable)  # 실행 파일의 디렉터리
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))  # 스크립트 경로

# 프로그래스 바 함수
def show_individual_progress_bar(message, duration=2):
    """각 작업에 대해 0%부터 100%까지 프로그래스 바를 표시"""
    total_steps = 100
    for step in range(1, total_steps + 1):
        percentage = step
        bar = f"[{'#' * (percentage // 5)}{'.' * (20 - (percentage // 5))}]"
        sys.stdout.write(f"\r{message}: {bar} {percentage}%")
        sys.stdout.flush()
        time.sleep(duration / total_steps)  # 총 작업 시간을 조절
    print("")  # 작업 완료 후 줄바꿈

# 모델 초기화
show_individual_progress_bar("Initializing device")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

show_individual_progress_bar("Loading model")
model_id = os.path.join(current_dir, "fine_tuned_whisper")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

show_individual_progress_bar("Loading processor")
processor = AutoProcessor.from_pretrained(model_id)

show_individual_progress_bar("Setting up pipeline")
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

show_individual_progress_bar("Finalizing setup")
generate_kwargs = {
    "language": "korean",
    "max_new_tokens": 100,
    "num_beams": 3,
    "condition_on_prev_tokens": True,
    "compression_ratio_threshold": 2.4,
    "temperature": 0.2,
    "logprob_threshold": 1.0,
    "no_speech_threshold": 100.0,
    "return_timestamps": True,
}

print("Initialization complete!")

# JSON 매핑 파일 로드
def resource_path(relative_path):
    """PyInstaller로 빌드된 실행 파일에서 데이터를 찾는 경로를 반환"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(__file__), relative_path)

with open(resource_path("units.json"), "r", encoding="utf-8") as f:
    units = json.load(f)

with open(resource_path("actions.json"), "r", encoding="utf-8") as f:
    actions = json.load(f)

# VAD 설정
vad = webrtcvad.Vad()
vad.set_mode(0)

# 오디오 스트림 설정
SAMPLE_RATE = 16000
FRAME_DURATION = 0.03
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)

# Unity와 소켓 통신 설정
HOST = "127.0.0.1"
PORT = 5005

def send_command_to_unity(command):
    """
    Unity로 명령어를 전송하는 함수
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect((HOST, PORT))
            s.sendall(command.encode('utf-8'))
            print(f"Sent to Unity: {command}\n")
        except ConnectionRefusedError:
            print("Unity 서버에 연결할 수 없습니다. Unity에서 서버를 실행 중인지 확인하세요.\n")

def normalize_text(text):
    """
    입력 텍스트를 정규화: 공백, 마침표, 쉼표 제거 및 영어 대문자는 소문자로
    """
    return text.replace(" ", "").replace(".", "").replace(",", "")

def find_command(normalized_text):
    """
    입력 텍스트에서 유닛과 동작을 찾는 함수
    """
    for unit_kor in units:
        if unit_kor in normalized_text:
            for action_kor in actions:
                if action_kor in normalized_text:
                    return units[unit_kor], actions[action_kor]
    return None, None

def frame_generator(audio_stream):
    """
    마이크로부터 고정된 길이의 오디오 프레임 생성.
    """
    while True:
        data, _ = audio_stream.read(FRAME_SIZE)
        audio = data[:, 0]
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()
        yield audio_bytes

def vad_collector(audio_stream):
    """
    프레임별로 VAD를 실행하고, 음성이 감지된 구간을 수집.
    """
    audio_buffer = b""
    for frame in frame_generator(audio_stream):
        if vad.is_speech(frame, SAMPLE_RATE):
            audio_buffer += frame
        elif audio_buffer:
            yield np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = b""

def main():
    print("Listening for audio... Press Ctrl+C to stop.")
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            unwanted_phrases = ["감사합니다", "네", "알겠습니다", "뒤로", "후퇴해", "아"]
            combined_pattern = re.compile(r"^((뒤로|후퇴해)(\s*|$)){2,}|(.)\4{3,}")
            for audio_segment in vad_collector(stream):
                audio_segment_resampled = resample(audio_segment, len(audio_segment) * SAMPLE_RATE // len(audio_segment))
                audio_input = {"array": audio_segment_resampled, "sampling_rate": SAMPLE_RATE}

                result = pipe(inputs=audio_input, generate_kwargs=generate_kwargs)
                transcription = result["text"]

                normalized_text = normalize_text(result["text"])
                unit_eng, action_eng = find_command(normalized_text)
                
                if (not unit_eng or not action_eng) and (
                normalized_text in unwanted_phrases or combined_pattern.match(normalized_text)):
                    continue

                if unit_eng and action_eng:
                    command = f"{unit_eng}|{action_eng}"
                    print("Transcription:", result["text"])
                    send_command_to_unity(command)
                else:
                    print("Transcription:", result["text"])
                    print("No valid command found in the transcription.\n")

    except KeyboardInterrupt:
        print("Program stopped.")

# 실행
if __name__ == "__main__":
    main()
