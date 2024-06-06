import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

# 곡선을 그리는 함수
def plot(category):
    if category == 'Loss':
        train = train_losses
        val = val_losses
    elif category == 'Accuracy':
        train = train_accuracies
        val = val_accuracies
    elif category == 'Precision':
        train = train_precisions
        val = val_precisions
    elif category == 'Recall':
        train = train_recalls
        val = val_recalls
    elif category == 'F1 Score':
        train = train_f1s
        val = val_f1s
    plt.figure()
    plt.plot(range(1, len(train) + 1), train, label=f'Training {category}')
    plt.plot(range(1, len(val) + 1), val, label=f'Validation {category}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{category}')
    plt.title(f'{category} Curve')
    plt.legend()
    plt.savefig(f'img/GRU_{category}_curve.png')

if __name__ == "__main__":
    # 데이터 로드
    voicephishing_path = 'data/augmented_transcriptions.xlsx'
    ecommerce_path = 'data/combined_transcriptions.xlsx'

    voicephishing_data = pd.read_excel(voicephishing_path)
    ecommerce_data = pd.read_excel(ecommerce_path)

    # 데이터에 라벨 추가
    voicephishing_data['label'] = 1
    ecommerce_data['label'] = 0

    # 데이터 합치기
    combined_data = pd.concat([voicephishing_data, ecommerce_data.sample(n=3000, random_state=42)])

    # 텍스트 전처리
    sentences = combined_data['transcript'].astype(str).tolist()
    labels = combined_data['label'].tolist()

    # 토큰화 및 패딩
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, padding='post')

    # 훈련 및 검증 데이터셋 분할
    x_train, temp_sequences, y_train, temp_labels = train_test_split(
        padded_sequences, labels, test_size=0.4, random_state=42)
    x_validation, x_test, y_validation, y_test = train_test_split(
        temp_sequences, temp_labels, test_size=0.5, random_state=42)
    
    # GRU 모델 정의
    vocab_size = len(word_index) + 1
    embedding_dim = 128
    max_length = padded_sequences.shape[1]

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        GRU(128, return_sequences=True),
        Dropout(0.5),
        GRU(128),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # 이진 분류
    ])

    # 모델 컴파일
    model.compile(loss='binary_crossentropy',  # 이진 분류
                  optimizer='adam',
                  metrics=['accuracy'])

    # ModelCheckpoint 콜백 설정
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_loss', verbose=1, save_best_only=True)

    # 모델 학습
    history = model.fit(x_train, np.array(y_train), epochs=50,
                        validation_data=(x_validation, np.array(y_validation)),
                        batch_size=32, verbose=1, callbacks=[checkpoint])

    # 모델 저장
    model.save('model/GRU_model.h5')

    # 학습 및 검증 손실, 정확도 저장
    train_accuracies = history.history['accuracy']
    val_accuracies = history.history['val_accuracy']
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    # 학습 및 검증 Precision, Recall, F1 Score 저장
    # Precision, Recall, F1 Score는 콜백으로 직접 추적되지 않으므로, 여기서는 나중에 평가
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []

    # 곡선 그리기
    plot('Loss')
    plot('Accuracy')

    # 모델 평가
    test_loss, test_accuracy = model.evaluate(np.array(x_test), np.array(y_test), verbose=2)

    # 모델 예측
    predictions = model.predict(np.array(x_test))
    predictions = (predictions > 0.5).astype(int)

    # 정밀도, 재현율, F1 스코어 계산
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    # 결과 출력
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Loss: {test_loss}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
