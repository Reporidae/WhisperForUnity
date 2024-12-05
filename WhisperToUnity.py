import os
import socket
import torch
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
    "language": "korean",             # 한국어 강제 설정
    "max_new_tokens": 100,            # 최대 토큰 수
    "num_beams": 3,                   # 빔 검색 수
    "condition_on_prev_tokens": True,  # 이전 토큰 조건 무시
    "compression_ratio_threshold": 2.4,  # 압축 비율 임계값
    "temperature": 0.2,
    "logprob_threshold": 1.0,        # 로그 확률 임계값
    "no_speech_threshold": 100.0,       # 무음 임계값
    "return_timestamps": True,        # 타임스탬프 반환
}

# VAD 설정
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0~3 (3이 가장 민감)

# 오디오 스트림 설정
SAMPLE_RATE = 16000  # Whisper 모델에 적합한 샘플링 레이트
FRAME_DURATION = 0.03  # 프레임 길이 (초), 10ms, 20ms, 30ms 중 하나
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)  # 프레임 크기

# Unity와 소켓 통신 설정
HOST = "127.0.0.1"  # Unity와 통신할 호스트 IP
PORT = 5005         # Unity와 통신할 포트

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
        # 마이크에서 고정된 길이의 데이터를 읽음
        data, _ = audio_stream.read(FRAME_SIZE)
        audio = data[:, 0]  # 1채널만 사용 (모노)
        audio_bytes = (audio * 32768).astype(np.int16).tobytes()  # 16-bit PCM 변환
        yield audio_bytes

def vad_collector(audio_stream):
    """
    프레임별로 VAD를 실행하고, 음성이 감지된 구간을 수집.
    """
    audio_buffer = b""
    for frame in frame_generator(audio_stream):
        # VAD로 음성 감지
        if vad.is_speech(frame, SAMPLE_RATE):
            audio_buffer += frame
        elif audio_buffer:
            # 음성 감지가 끝나면 버퍼 반환
            yield np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
            audio_buffer = b""  # 버퍼 초기화

def main():
    print("Listening for audio... Press Ctrl+C to stop.")
    try:
        # 마이크에서 오디오 스트림 열기
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32') as stream:
            for audio_segment in vad_collector(stream):
                print("Processing detected speech...")
                # Whisper 입력 크기에 맞게 재샘플링
                audio_segment_resampled = resample(audio_segment, len(audio_segment) * SAMPLE_RATE // len(audio_segment))
                # Whisper 입력 형식으로 변환
                audio_input = {"array": audio_segment_resampled, "sampling_rate": SAMPLE_RATE}


                # 텍스트 변환
                result = pipe(inputs=audio_input , generate_kwargs=generate_kwargs)
                #result = pipe(inputs=audio_input)
                print("Transcription:", result["text"])

                # 텍스트 정규화
                normalized_text = normalize_text(result["text"])
                print(f"Normalized Text: {normalized_text}")

                # 유닛 및 동작 매핑
                units = {
                    "전사": "SWORDMAN",
                    "창병": "SPEARMAN",
                    "장병": "SPEARMAN",
                    "궁수": "ARCHER",
                    "공수": "ARCHER",
                    "공소": "ARCHER",# 오탈자도 알맞게 매핑시킴
                    "궁서": "ARCHER",
                    "방패병": "SHIELDMAN",
                    "검사": "SWORDMAN"
                }

                actions = {
                    "앞": "MOVE_FORWARD",
                    "전진" : "MOVE_FORWARD",
                    "뒤": "MOVE_BACKWARD",
                    "왼": "MOVE_LEFT",
                    "오른": "MOVE_RIGHT",
                    "후퇴": "RETREAT"
                }

                # 명령어 분석 및 전송
                for unit_kor, unit_eng in units.items():
                    for action_kor, action_eng in actions.items():
                        if f"{unit_kor}{action_kor}" in normalized_text:
                            send_command_to_unity(f"{unit_eng}|{action_eng}")
                            print("{unit_eng}|{action_eng}")

    except KeyboardInterrupt:
        print("Program stopped.")

# 실행
if __name__ == "__main__":
    main()
