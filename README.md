# Voice-Phishing-Detector

# 코드 설명


## voicephishing_preprocessing.py

이 스크립트는 다음과 같은 작업을 수행합니다:

1. **기본 설정**: Google Cloud 자격 증명을 설정하고 필요한 라이브러리를 불러옵니다.
2. **오디오 파일 업로드**: `upload_blob` 함수를 사용하여 Google Cloud Storage에 오디오 파일을 업로드합니다.
3. **오디오 파일 변환**:
    - `transcribe_long_audio`: 긴 오디오 파일을 텍스트로 변환합니다.
    - `transcribe_speech_short`: 짧은 오디오 파일을 텍스트로 변환합니다.
4. **텍스트 데이터 증강**:
    - `synonym_replacement`: 문장에서 n개의 단어를 동의어로 대체합니다.
    - `random_insertion`: 문장에 n개의 단어를 무작위로 삽입합니다.
    - `random_deletion`: 문장에서 확률 p로 단어를 무작위로 삭제합니다.
5. **데이터 저장**: 변환된 텍스트와 증강된 데이터를 CSV 및 Excel 파일로 저장합니다.

### 코드 세부 설명

- **upload_blob(bucket_name, source_file_name, destination_blob_name)**: 지정된 버킷에 파일을 업로드합니다.
- **transcribe_long_audio(gcs_uri)**: Google Cloud Storage의 긴 오디오 파일을 텍스트로 변환합니다.
- **transcribe_speech_short(audio_file_path)**: 짧은 오디오 파일을 텍스트로 변환합니다.
- **synonym_replacement(sentence, n)**: 문장에서 n개의 단어를 동의어로 대체합니다.
- **get_synonyms(word)**: 주어진 단어의 동의어를 가져옵니다.
- **random_insertion(sentence, n)**: 문장에 무작위로 n개의 단어를 삽입합니다.
- **add_word(new_words)**: 단어 목록에 단어를 추가하는 보조 함수입니다.
- **random_deletion(sentence, p)**: 문장에서 확률 p로 단어를 무작위로 삭제합니다.

### 사용법

#### Google Cloud 설정

1. Google Cloud Console에서 프로젝트를 생성하고, Speech-to-Text API와 Storage API를 활성화합니다.
2. 서비스 계정 키 파일(JSON)을 생성하여 다운로드합니다.

   
#### 오디오 파일 업로드

Google Cloud Storage에 오디오 파일을 업로드하려면 `upload_blob` 함수를 사용합니다.

```python
upload_blob(bucket_name="your-bucket-name", source_file_name="path/to/your/audiofile.mp3", destination_blob_name="audiofiles/audiofile.mp3")
```

#### 오디오 파일 변환

오디오 파일을 텍스트로 변환하려면 `transcribe_long_audio` 함수를 사용합니다.

```python
response = transcribe_long_audio(gcs_uri="gs://your-bucket-name/audiofiles/audiofile.mp3")
transcript = ''.join(result.alternatives[0].transcript for result in response.results)
```

#### 텍스트 데이터 증강

변환된 텍스트 데이터를 증강하려면 다음 함수를 사용합니다:

- `synonym_replacement(sentence, n)`: 문장에서 n개의 단어를 동의어로 대체합니다.
- `random_insertion(sentence, n)`: 문장에 n개의 단어를 무작위로 삽입합니다.
- `random_deletion(sentence, p)`: 문장에서 확률 p로 단어를 무작위로 삭제합니다.

```python
augmented_data = []
augmented_data.append({'transcript': transcript})
augmented_data.append({'transcript': synonym_replacement(transcript, 3)})
augmented_data.append({'transcript': random_insertion(transcript, 3)})
augmented_data.append({'transcript': random_deletion(transcript, 0.3)})
```

#### 데이터 저장

증강된 데이터를 CSV 또는 Excel 파일로 저장합니다.

```python
augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_excel('data/augmented_transcriptions.xlsx', index=False)
```

### 설치

이 모듈을 실행하기 위해서는 다음의 도구와 라이브러리가 필요합니다:

- Python 3.x
- Google Cloud SDK
- pandas
- nltk
- google-cloud-speech
- google-cloud-storage
- openpyxl

필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```sh
pip install pandas nltk google-cloud-speech google-cloud-storage openpyxl
```



## call_preprocessing.py

이 스크립트는 다음과 같은 작업을 수행합니다:

1. **기본 경로 설정**: 데이터가 저장된 기본 폴더 경로를 설정합니다.
2. **폴더 번호 범위 설정**: 처리할 폴더의 시작 번호와 끝 번호를 설정합니다.
3. **폴더 순회 및 JSON 파일 처리**:
    - 설정된 범위 내의 모든 폴더를 순회합니다.
    - 각 폴더 내의 JSON 파일을 확인하고 존재하는 경우 파일을 읽어들입니다.
    - JSON 파일 내의 대화 데이터를 결합하여 하나의 텍스트로 만듭니다.
    - 결합된 텍스트에서 한글 문자와 공백만을 추출합니다.
    - 추출된 한글 텍스트와 파일 이름을 데이터 리스트에 저장합니다.
4. **데이터프레임 생성**: 모든 처리된 데이터를 하나의 데이터프레임으로 결합합니다.
5. **엑셀 파일로 저장**: 데이터프레임을 엑셀 파일로 저장합니다.

### 코드 세부 설명

- **extract_korean(text)**: 주어진 텍스트에서 한글 문자와 공백만 남기고 나머지 문자를 제거하는 함수입니다.
- **파일 존재 여부 확인 및 데이터 처리**: 각 폴더 내에 JSON 파일이 존재하는지 확인하고, 존재할 경우 파일을 열어 데이터를 처리합니다. 처리 중 오류가 발생하면 오류 메시지를 출력합니다.
- **데이터프레임 생성 및 저장**: 처리된 데이터를 데이터프레임으로 변환하고, 이를 엑셀 파일로 저장합니다.

### 사용법

1. **Python 환경 설정**: pandas와 re 모듈이 필요합니다. 필요시 다음 명령어로 설치할 수 있습니다.
    ```sh
    pip install pandas
    ```

2. **기본 경로 및 범위 설정**: `base_path`, `start_index`, `end_index` 변수를 필요한 값으로 설정합니다.

3. **스크립트 실행**:
    ```sh
    python call_preprocessing.py
    ```

4. **결과 확인**: 스크립트 실행 후 `data/combined_transcriptions.xlsx` 파일이 생성되며, 결합된 텍스트 데이터가 저장됩니다.



## GNN.py

이 스크립트는 다음과 같은 작업을 수행합니다:

1. **데이터 로드 및 전처리**:
    - 음성 피싱 및 전자상거래 데이터를 불러와서 라벨을 추가합니다.
    - 두 데이터를 합치고, TF-IDF 벡터화를 수행하여 피처 행렬을 생성합니다.
2. **그래프 생성**:
    - TF-IDF 벡터를 기반으로 그래프를 생성하고 시각화합니다.
3. **GNN 모델 정의**:
    - 그래프 컨볼루션 네트워크(GCN)를 사용하여 모델을 정의합니다.
4. **훈련 및 평가**:
    - 모델을 훈련하고, 검증 및 테스트 데이터를 사용하여 평가합니다.
    - 훈련 과정 중 손실, 정확도, 정밀도, 재현율, F1 점수를 기록하고 시각화합니다.
5. **모델 저장 및 결과 출력**:
    - 최종 모델을 저장하고, 훈련 및 테스트 결과를 출력합니다.

### 코드 세부 설명

- **build_graph(X, y)**: TF-IDF 벡터를 기반으로 그래프를 생성합니다.
- **GNN**: PyTorch Geometric의 GCNConv를 사용하여 정의된 GNN 모델 클래스입니다.
- **train()**: 모델을 훈련시키는 함수입니다.
- **evaluate(mask)**: 주어진 데이터 마스크(train, val, test)를 사용하여 모델을 평가하는 함수입니다.
- **plot(category)**: 훈련 및 검증 곡선을 시각화하여 저장하는 함수입니다.

### 설치

이 프로젝트를 실행하기 위해서는 다음의 도구와 라이브러리가 필요합니다:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- torch
- torch-geometric
- networkx

필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```sh
pip install pandas numpy scikit-learn matplotlib torch torch-geometric networkx
```

### 사용법

1. **데이터 준비**: 
    - `data/augmented_transcriptions.xlsx`와 `data/combined_transcriptions.xlsx` 파일을 준비합니다.
    - `data/token_graph.gml` 파일이 있는지 확인합니다. 없다면 코드가 그래프를 생성하고 저장합니다.

2. **스크립트 실행**:
    ```sh
    python GNN.py
    ```

3. **결과 확인**: 
    - 훈련 및 검증 곡선은 `img/` 폴더에 저장됩니다.
    - 최종 모델은 `model/gnn_model.pth`로 저장됩니다.
    - 그래프 시각화는 `img/graph_visualization.png`로 저장됩니다.
    - 콘솔에서 훈련 및 테스트 결과를 확인할 수 있습니다.


## Conv1D+BILSTM.py

이 스크립트는 다음과 같은 작업을 수행합니다:

1. **데이터 로드 및 전처리**:
    - 음성 피싱 및 전자상거래 데이터를 불러와서 라벨을 추가합니다.
    - 두 데이터를 합치고, TF-IDF 벡터화를 수행하여 피처 행렬을 생성합니다.
2. **데이터 임베딩**:
    - Word2Vec 모델을 사용하여 텍스트 데이터를 임베딩합니다.
    - 임베딩된 데이터를 배치 단위로 처리하고 저장합니다.
3. **모델 정의 및 학습**:
    - Conv1D와 Bidirectional LSTM을 포함한 모델 아키텍처를 정의합니다.
    - EarlyStopping과 ModelCheckpoint 콜백을 사용하여 모델을 학습합니다.
4. **모델 평가 및 저장**:
    - 학습된 모델을 평가하고, 정확도, 정밀도, 재현율, F1 점수를 출력합니다.
    - 최종 모델을 저장합니다.
5. **훈련 및 검증 곡선 시각화**:
    - 학습 과정에서의 손실, 정확도, 정밀도, 재현율, F1 점수를 시각화하여 저장합니다.

### 코드 세부 설명

- **clean_text(text)**: 주어진 텍스트에서 이상한 문자를 제거합니다.
- **batch_process_embedding(sequences, embedding_matrix, batch_size, output_dir)**: 배치 단위로 임베딩 처리를 수행합니다.
- **load_and_concatenate_batches(output_dir)**: 저장된 배치 파일들을 로드하여 하나의 데이터로 결합합니다.
- **print_class_distribution(y, dataset_name)**: 클래스 분포를 출력합니다.
- **AttentionLayer**: 커스텀 어텐션 레이어를 정의합니다.
- **plot(category)**: 학습 및 검증 곡선을 시각화하여 저장합니다.

### 설치

이 프로젝트를 실행하기 위해서는 다음의 도구와 라이브러리가 필요합니다:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow
- keras
- gensim
- imbalanced-learn

필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```sh
pip install pandas numpy scikit-learn matplotlib tensorflow keras gensim imbalanced-learn
```

### 사용법

1. **데이터 준비**:
    - `data/augmented_transcriptions.xlsx`와 `data/combined_transcriptions.xlsx` 파일을 준비합니다.

2. **스크립트 실행**:
    ```sh
    python Conv1D+BILSTM.py
    ```

3. **결과 확인**:
    - 훈련 및 검증 곡선은 `img/` 폴더에 저장됩니다.
    - 최종 모델은 `model/BILSTM_model.h5`로 저장됩니다.
    - 콘솔에서 훈련 및 테스트 결과를 확인할 수 있습니다.


## koSBERT.py

이 스크립트는 다음과 같은 작업을 수행합니다:

1. **데이터 로드 및 전처리**:
    - 음성 피싱 및 전자상거래 데이터를 불러와서 라벨을 추가합니다.
    - 두 데이터를 합치고, 텍스트 데이터를 전처리하여 임베딩 벡터로 변환합니다.
2. **모델 정의 및 학습**:
    - KoSimCSE 모델을 사용하여 텍스트 데이터를 임베딩합니다.
    - 분류 모델 아키텍처를 정의합니다.
    - 모델을 학습하고, 학습 과정에서의 손실, 정확도, 정밀도, 재현율, F1 점수를 기록합니다.
3. **모델 평가 및 저장**:
    - 학습된 모델을 평가하고, 최종 결과를 출력합니다.
    - 최종 모델을 저장합니다.
4. **훈련 및 검증 곡선 시각화**:
    - 학습 과정에서의 손실, 정확도, 정밀도, 재현율, F1 점수를 시각화하여 저장합니다.

### 코드 세부 설명

- **CustomDataset**: 사용자 정의 데이터셋 클래스로, 데이터 로딩을 위한 클래스입니다.
- **ClassificationModel**: 분류 모델 클래스입니다. 여러 층의 Fully Connected 레이어와 Dropout, Batch Normalization을 포함합니다.
- **evaluate**: 모델을 평가하는 함수입니다.
- **plot**: 학습 곡선을 그리는 함수입니다.

### 설치

이 프로젝트를 실행하기 위해서는 다음의 도구와 라이브러리가 필요합니다:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- torch
- transformers
- tqdm

필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```sh
pip install pandas numpy scikit-learn matplotlib torch transformers tqdm
```

### 사용법

1. **모델과 토크나이저 준비**:
    - 로컬에 `KoSentenceBERT_SKTBERT` 저장소를 클론합니다:
    ```sh
    git clone https://github.com/BM-K/KoSentenceBERT_SKTBERT.git
    ```
    - 모델과 토크나이저를 로컬 경로에서 불러옵니다.

2. **데이터 준비**:
    - `data/augmented_transcriptions.xlsx`와 `data/combined_transcriptions.xlsx` 파일을 준비합니다.

3. **스크립트 실행**:
    ```sh
    python koSBERT.py
    ```

4. **결과 확인**:
    - 훈련 및 검증 곡선은 `img/` 폴더에 저장됩니다.
    - 최종 모델은 `final_model.pth`로 저장됩니다.
    - 콘솔에서 훈련 및 테스트 결과를 확인할 수 있습니다.



## GRU.py

이 스크립트는 다음과 같은 작업을 수행합니다:

1. **데이터 로드 및 전처리**:
    - 음성 피싱 및 전자상거래 데이터를 불러와서 라벨을 추가합니다.
    - 두 데이터를 합치고, 텍스트 데이터를 전처리하여 패딩된 시퀀스로 변환합니다.
2. **모델 정의 및 학습**:
    - GRU 모델 아키텍처를 정의합니다.
    - 모델을 학습하고, 학습 과정에서의 손실, 정확도, 정밀도, 재현율, F1 점수를 기록합니다.
3. **모델 평가 및 저장**:
    - 학습된 모델을 평가하고, 최종 결과를 출력합니다.
    - 최종 모델을 저장합니다.
4. **훈련 및 검증 곡선 시각화**:
    - 학습 과정에서의 손실, 정확도, 정밀도, 재현율, F1 점수를 시각화하여 저장합니다.

### 코드 세부 설명

- **plot**: 학습 곡선을 그리는 함수입니다.
- **CustomDataset**: 사용자 정의 데이터셋 클래스로, 데이터 로딩을 위한 클래스입니다.
- **ClassificationModel**: 분류 모델 클래스입니다. 여러 층의 Fully Connected 레이어와 Dropout, Batch Normalization을 포함합니다.
- **evaluate**: 모델을 평가하는 함수입니다.

### 설치

이 프로젝트를 실행하기 위해서는 다음의 도구와 라이브러리가 필요합니다:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- tensorflow
- keras
- tqdm

필요한 라이브러리를 설치하려면 다음 명령어를 사용하세요:

```sh
pip install pandas numpy scikit-learn matplotlib tensorflow keras tqdm
```

### 사용법

1. **데이터 준비**:
    - `data/augmented_transcriptions.xlsx`와 `data/combined_transcriptions.xlsx` 파일을 준비합니다.

2. **스크립트 실행**:
    ```sh
    python GRU.py
    ```

3. **결과 확인**:
    - 훈련 및 검증 곡선은 `img/` 폴더에 저장됩니다.
    - 최종 모델은 `model/GRU_model.h5`로 저장됩니다.
    - 콘솔에서 훈련 및 테스트 결과를 확인할 수 있습니다.


