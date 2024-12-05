import os
import socket
import torch
import json
import re
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import webrtcvad
import numpy as np
from scipy.signal import resample

# 모델 초기화
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# 현재 스크립트의 디렉터리 경로 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# 파인 튜닝된 모델의 상대 경로 설정
model_id = os.path.join(current_dir, "fine_tuned_whisper")
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

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

# JSON 매핑 파일 로드
with open(os.path.join(current_dir, "units.json"), "r", encoding="utf-8") as f:
    units = json.load(f)

with open(os.path.join(current_dir, "actions.json"), "r", encoding="utf-8") as f:
    actions = json.load(f)

# 정규 표현식 패턴 생성
pattern = r"(" + "|".join(units.keys()) + r")(" + "|".join(actions.keys()) + r")"
regex = re.compile(pattern)

# VAD 설정
vad = webrtcvad.Vad()
vad.set_mode(3)

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
            print(f"Sent to Unity: {command}")
        except ConnectionRefusedError:
            print("Unity 서버에 연결할 수 없습니다. Unity에서 서버를 실행 중인지 확인하세요.")

def normalize_text(text):
    """
    입력 텍스트를 정규화: 공백 제거 및 소문자 변환
    """
    return text.replace(" ", "").strip()

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
            for audio_segment in vad_collector(stream):
                print("Processing detected speech...")
                audio_segment_resampled = resample(audio_segment, len(audio_segment) * SAMPLE_RATE // len(audio_segment))
                audio_input = {"array": audio_segment_resampled, "sampling_rate": SAMPLE_RATE}

                result = pipe(inputs=audio_input, generate_kwargs=generate_kwargs)
                print("Transcription:", result["text"])

                normalized_text = normalize_text(result["text"])
                print(f"Normalized Text: {normalized_text}")

                # 정규 표현식으로 명령어 탐색 후 유니티로 전송
                match = regex.search(normalized_text)
                if match:
                    unit_kor = match.group(1)  # 첫 번째 그룹: 유닛
                    action_kor = match.group(2)  # 두 번째 그룹: 동작

                    unit_eng = units[unit_kor]
                    action_eng = actions[action_kor]
                    command = f"{unit_eng}|{action_eng}"
                    send_command_to_unity(command)
                    print(f"Processed Command: {command}")
                else:
                    print("No valid command found in the transcription.")


    except KeyboardInterrupt:
        print("Program stopped.")

# 실행
if __name__ == "__main__":
    main()
