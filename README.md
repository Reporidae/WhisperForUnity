# WhisperForUnity
Whisper에서 Unity로 음성 인식 텍스트를 전달하는 프로젝트.

---

## 사용법

### 1. 프로젝트 사용법

1. **다음 파일들이 같은 경로에 있어야 함**
   - 위스퍼 모델 폴더 (WhisperToUnity.py 코드의 model_id 부분은 폴더 이름과 똑같이 수정 필요)
   - WhisperToUnity.py
   - actions.json
   - units.json

2. **Unity 설정**:
   - Unity 스크립트 폴더에 다음 스크립트 추가:
     - `CommandProcessor`
     - `MovableObject`
     - `NetworkManager`
     - `ObjectController`
   - 빈 게임 오브젝트를 생성하고, 컴포넌트에 `ObjectController` 추가.
   - 3D 오브젝트를 생성하고 다음과 같이 이름 지정:
     - `SHIELDMAN`
     - `SWORDMAN`
     - `ARHCER`
   - 각 오브젝트에 `MovableObject` 스크립트를 추가.
3. **실행**:
   - Unity 게임 실행.
   - Python Whisper 서버 실행.
4. **테스트**:
   - 마이크에 다음 명령어를 말해 테스트:
     - **"방패병 앞으로 가"**
     - **"전사 뒤로 가"**
     - **"궁수 왼쪽으로 가"**
5. **오브젝트 이름이나 명령어를 수정하고 싶으면 units.json과 actions.json 파일을 먼저 수정한 뒤 CommandProcessor.cs 파일에 switch(actions) case문 쪽을 그에 맞게 수정해줌**

---

### 2. WhisperFineTuning 사용법

1. **폴더 만들기**
   - 이 스크립트가 위치한 폴더에 dataset 폴더를 만든다. 
   - dataset 폴더 안에 train과 val 폴더를 각각 만든다.
2. **데이터 준비하기**:
   - train 폴더엔 학습시킬 .wav 형식의 오디오 파일을 넣는다.
   - val 폴더엔 학습시킨 결과를 검증할 .wav 형식의 오디오 파일을 넣는다.
   - 엑셀로 각각 폴더에 .csv 형식의 파일을 작성한다. 맨 첫 줄엔 각각 filename과 text를 적고 filename엔 오디오 파일 이름, text엔 그 오디오 파일이 나타내는 텍스트를 입력한다.
     - | filename | text |
       |-----------|-----------|
       |audio1.wav|궁수 앞으로 가 |
       |audio2.wav|궁수 뒤로 가   |
       |audio3.wav|궁수  가      |
       | ...      | ...         |
       
3. 코드를 실행하고 기다린다. 학습이 끝나면 스크립트가 위치한 폴더에 파인 튜닝된 위스퍼 모델의 폴더가 생성됐는지 확인한다.

---

### 3. 파이썬 - 유니티 통합 방법

1. **파이썬 파일 빌드**
   - pyinstaller를 이용해서 파이썬 코드를 .exe 실행 파일로 빌드한다.
     - 안 깔려 있다면 터미널에 pip install pyinstaller를 입력해서 설치한다.
     - 사용 예시:
       ```bash
       pyinstaller --onefile --add-data "actions.json;." --add-data "units.json;." WhisperToUnity.py
2. **파일 경로 매칭**
   - 빌드한 파일을 유니티 Assets - StreamingAssets - Python 경로에 넣는다. (원래 처음엔 없으니 만들어야 함 경로를 다르게 바꾸고 싶으면 PythonLauncher 코드 수정)
   - 위스퍼 모델 폴더도 같은 경로에 넣는다.
3. **유니티로 파이썬 파일 실행**
   - 빈 게임 오브젝트를 만들고 PythonLauncher 코드를 컴포넌트로 추가시킨다. 그럼 유니티에서 게임을 실행하면 파이썬 .exe 파일도 같이 실행된다.
4. **빌드**
   - 유니티의 빌드 버튼을 눌러 최종 게임 파일로 빌드한다.

---

## 전제 조건

Whisper large v3 turbo 모델이 설치된 아나콘다 가상 환경 서버에서 코드를 실행해야 합니다. 이를 위해 아래 프로그램들이 설치되어야 합니다:

- **Python**: [3.12.7](https://www.python.org/downloads/release/python-3127/)
- **PyTorch**: [Stable 2.5.1, CUDA 12.1](https://pytorch.kr/get-started/locally/)
- **CUDA Toolkit**: [12.1 Update 1](https://developer.nvidia.com/cuda-12-1-1-download-archive)
- **cuDNN**: [v8.9.7 for CUDA 12.x](https://developer.nvidia.com/rdp/cudnn-archive)

추가 패키지:
- `sounddevice`
- `webrtcvad`

위 두 패키지는 `pip` 명령어를 사용해 설치 가능합니다:
```bash
pip install sounddevice webrtcvad
