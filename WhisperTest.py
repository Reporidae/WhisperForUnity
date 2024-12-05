import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import sounddevice as sd
import webrtcvad
import numpy as np
from scipy.signal import resample

# 모델 초기화
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
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

# VAD 설정
vad = webrtcvad.Vad()
vad.set_mode(3)  # 0~3 (3이 가장 민감)

# 오디오 스트림 설정
SAMPLE_RATE = 16000  # Whisper 모델에 적합한 샘플링 레이트
FRAME_DURATION = 0.02  # 프레임 길이 (초), 10ms, 20ms, 30ms 중 하나
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION)  # 프레임 크기

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
                result = pipe(audio_input)
                print("Transcription:", result["text"])
    except KeyboardInterrupt:
        print("Program stopped.")

# 실행
if __name__ == "__main__":
    main()
