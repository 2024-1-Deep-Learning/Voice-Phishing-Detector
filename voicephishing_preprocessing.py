import os
import io
import random
import pandas as pd
from google.cloud import speech, storage
import nltk
from nltk.corpus import wordnet

# Google Cloud Storage 버킷에 파일을 업로드하는 함수
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """버킷에 파일을 업로드합니다."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    print(f"파일 {source_file_name}이(가) {destination_blob_name}에 업로드되었습니다.")

# Google Cloud Speech-to-Text API를 사용하여 긴 오디오 파일을 텍스트로 변환하는 함수
def transcribe_long_audio(gcs_uri):
    """Google Cloud Storage에서 긴 오디오 파일을 텍스트로 변환합니다."""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        max_alternatives=1,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    operation = client.long_running_recognize(config=config, audio=audio)
    print(f"{gcs_uri} 변환 중...")
    response = operation.result(timeout=900)  # 최대 15분 대기
    return response

# Google Cloud Speech-to-Text API를 사용하여 짧은 오디오 파일을 텍스트로 변환하는 함수
def transcribe_speech_short(audio_file_path):
    """짧은 오디오 파일을 텍스트로 변환합니다."""
    client = speech.SpeechClient()
    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,
        sample_rate_hertz=16000,
        max_alternatives=1,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )
    print("작업 완료 대기 중...")
    response = client.recognize(config=config, audio=audio)
    return response

# 문장에서 단어를 동의어로 대체하는 함수
def synonym_replacement(sentence, n):
    """문장에서 n개의 단어를 동의어로 대체합니다."""
    words = sentence.split()
    if not words:
        return sentence
    new_words = words.copy()
    random_word_list = list(set([word for word in words if wordnet.synsets(word)]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if synonyms:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break
    return ' '.join(new_words)

# 단어의 동의어를 가져오는 함수
def get_synonyms(word):
    """주어진 단어의 동의어를 가져옵니다."""
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonyms.discard(word)
    return list(synonyms)

# 문장에 무작위로 단어를 삽입하는 함수
def random_insertion(sentence, n):
    """문장에 무작위로 n개의 단어를 삽입합니다."""
    words = sentence.split()
    if not words:
        return sentence
    new_words = words.copy()
    for _ in range(n):
        add_word(new_words)
    return ' '.join(new_words)

def add_word(new_words):
    """단어 목록에 단어를 추가하는 보조 함수."""
    synonyms = []
    counter = 0
    while not synonyms and counter < 10:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = get_synonyms(random_word)
        counter += 1
    if synonyms:
        random_synonym = synonyms[0]
        random_idx = random.randint(0, len(new_words))
        new_words.insert(random_idx, random_synonym)

# 문장에서 단어를 무작위로 삭제하는 함수
def random_deletion(sentence, p):
    """문장에서 확률 p로 단어를 무작위로 삭제합니다."""
    words = sentence.split()
    if len(words) <= 1:
        return sentence
    new_words = [word for word in words if random.uniform(0, 1) > p]
    return ' '.join(new_words) if new_words else random.choice(words)

if __name__ == "__main__":
    # Google 자격 증명 설정
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'voice-phishing-detector-1c6017a2e15a.json'

    # Google Drive에 파일 업로드 필요 시 사용
    # drive.mount('/content/drive')

    # Google Cloud Storage에 파일을 업로드하려면 주석 해제
    # bucket_name = "your-bucket-name"
    # source_file_name = "path/to/your/audiofile.mp3"
    # destination_blob_name = "audiofiles/audiofile.mp3"
    # upload_blob(bucket_name, source_file_name, destination_blob_name)

    client = speech.SpeechClient()
    transcripts = []

    # 오디오 파일 변환
    for i in range(1, 411):
        gcs_uri = f"gs://voicephising_seyeon/voicephishing/{i:04}.mp3"
        response = transcribe_long_audio(gcs_uri)
        transcript = ''.join(result.alternatives[0].transcript for result in response.results)
        transcripts.append({'file_name': f"{i:03}.mp3", 'transcript': transcript})

    # 변환 결과를 데이터프레임으로 변환
    script_df = pd.DataFrame(transcripts, columns=['file_name', 'transcript'])
    script_df.to_csv('data/transcriptions.csv', index=False)
    print("변환 결과가 'transcriptions.csv' 파일에 저장되었습니다.")

    # NLTK wordnet 데이터 다운로드
    nltk.download('wordnet')

    # 변환 데이터 로드
    data_path = 'data/transcriptions.csv'
    data = pd.read_csv(data_path)

    # 결측값을 빈 문자열로 대체
    data['transcript'] = data['transcript'].fillna('')

    # 증강된 데이터 생성
    augmented_data = []

    for _, row in data.iterrows():
        original_sentence = row['transcript']
        augmented_data.append({'transcript': original_sentence})
        augmented_data.append({'transcript': synonym_replacement(original_sentence, 3)})
        augmented_data.append({'transcript': random_insertion(original_sentence, 3)})
        augmented_data.append({'transcript': random_deletion(original_sentence, 0.3)})

    augmented_df = pd.DataFrame(augmented_data)
    output_path = 'data/augmented_transcriptions.xlsx'
    augmented_df.to_excel(output_path, index=False)
    print(f"증강된 데이터가 {output_path} 파일에 저장되었습니다.")
