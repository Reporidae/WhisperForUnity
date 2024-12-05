import os
import librosa
import torch
from datasets import load_dataset
from transformers import WhisperFeatureExtractor, WhisperProcessor
from transformers import WhisperForConditionalGeneration, TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence

# 1. 상대 경로로 데이터셋 경로 설정
base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트 파일의 디렉토리
dataset_dir = os.path.join(base_dir, "dataset")  # dataset 폴더의 경로

# 데이터셋 파일 경로
train_metadata_path = os.path.join(dataset_dir, "train", "metadata.csv")
val_metadata_path = os.path.join(dataset_dir, "val", "metadata.csv")

# 2. 데이터셋 로드
print("Loading dataset...")
dataset = load_dataset("csv", data_files={
    "train": train_metadata_path,
    "val": val_metadata_path
})

# 3. 전처리 함수 정의
print("Preprocessing data...")
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3-turbo")
processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

def preprocess_function(examples, dataset_type):
    # 상대 경로로 오디오 파일 경로 설정
    audio_file = os.path.join(dataset_dir, dataset_type, examples["filename"])
    
    # 오디오 파일 로드
    audio, sampling_rate = librosa.load(audio_file, sr=16000)
    
    # Whisper 모델 전용 입력 데이터 생성
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt")
    inputs["labels"] = processor.tokenizer(examples["text"]).input_ids  # 텍스트를 labels로 변환
    
    return {
        "input_features": inputs.input_features.squeeze(0).numpy(),  # numpy로 변환
        "labels": inputs.labels  # 이미 Tensor 형식
    }

# train 데이터셋 처리
train_dataset = dataset["train"].map(
    lambda examples: preprocess_function(examples, "train"),
    remove_columns=["filename", "text"]
)

# val 데이터셋 처리
val_dataset = dataset["val"].map(
    lambda examples: preprocess_function(examples, "val"),
    remove_columns=["filename", "text"]
)

# 4. 모델 초기화
print("Initializing model...")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")

# 5. 데이터 Collator 정의
class WhisperDataCollator:
    def __call__(self, features):
        # input_features를 Tensor로 변환
        input_features = [torch.tensor(f["input_features"]) for f in features]
        input_features = pad_sequence(input_features, batch_first=True)
        
        # labels를 Tensor로 변환
        labels = [torch.tensor(f["labels"]) for f in features]
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return {"input_features": input_features, "labels": labels}

data_collator = WhisperDataCollator()

# 6. 학습 설정
print("Setting up training...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    save_steps=10,
    logging_dir="./logs",
    learning_rate=1e-5,
    fp16=True,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator    # 수정된 부분
)

# 7. 학습 시작
print("Starting training...")
trainer.train()

# 8. 학습된 모델 저장
print("Saving fine-tuned model...")
model.save_pretrained("fine_tuned_whisper")
processor.save_pretrained("fine_tuned_whisper")

print("Fine-tuning complete! Model saved in 'fine_tuned_whisper'.")
