from google.cloud import speech
from google.cloud import storage
import os
import io
import pandas as pd

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)
    print(f"File {source_file_name} uploaded to {destination_blob_name}.")


def transcribe_long_audio(gcs_uri):
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,  # 오디오 파일 형식에 맞게 설정
        sample_rate_hertz=16000,
        max_alternatives=1,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print(f"Transcribing {gcs_uri}...")
    response = operation.result(timeout=900)  # 최대 15분 대기

#     for result in response.results:
#         print("Transcript: {}".format(result.alternatives[0].transcript))
        
    return response

def transcribe_speech_short(audio_file_path):
    client = speech.SpeechClient()

    with io.open(audio_file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MP3,  # 오디오 파일 형식에 맞게 설정
        sample_rate_hertz=16000,
        max_alternatives=1,
        language_code="ko-KR",
        enable_automatic_punctuation=True,
    )

    print("Waiting for operation to complete...")
    response = client.recognize(config=config, audio=audio)

#     for result in response.results:
#         print("Transcript: {}".format(result.alternatives[0].transcript))
        
    return response

if __name__ == "__main__":
    # 업로드 필요시 사용
    # bucket_name = "your-bucket-name"
    # source_file_name = "path/to/your/audiofile.mp3"
    # destination_blob_name = "audiofiles/audiofile.mp3"

    # upload_blob(bucket_name, source_file_name, destination_blob_name)
    #setting Google credential
    os.environ['GOOGLE_APPLICATION_CREDENTIALS']= 'voice-phishing-detector-1c6017a2e15a.json'
    # create client instance 
    client = speech.SpeechClient()
    transcripts = []

    for i in range(1, 411):
        if i < 10:
            gcs_uri = f"gs://voicephising_seyeon/voicephishing/000{i}.mp3"
        elif i < 100:
            gcs_uri = f"gs://voicephising_seyeon/voicephishing/00{i}.mp3"
        else:
            gcs_uri = f"gs://voicephising_seyeon/voicephishing/0{i}.mp3"

        response = transcribe_long_audio(gcs_uri)
        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript

        row = {'file_name': f"{i:03}.mp3", 'transcript': transcript}
        transcripts.append(row)

    # 리스트를 데이터프레임으로 변환
    script_df = pd.DataFrame(transcripts, columns=['file_name', 'transcript'])

    # 데이터프레임을 CSV 파일로 저장
    script_df.to_csv('transcriptions.csv', index=False)
    print("Transcriptions have been saved to 'transcriptions.csv'")