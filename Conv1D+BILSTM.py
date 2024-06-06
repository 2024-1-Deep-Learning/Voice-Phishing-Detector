from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Layer, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# 텍스트에서 이상한 문자를 제거하는 함수
def clean_text(text):
    for char in weird_characters:
        text = text.replace(char, '')
    return text

# 배치 단위로 임베딩 처리하는 함수
def batch_process_embedding(sequences, embedding_matrix, batch_size=10000, output_dir='data/embedded_batches'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_sequences = sequences.shape[0]
    for start in range(0, num_sequences, batch_size):
        end = min(start + batch_size, num_sequences)
        batch_sequences = sequences[start:end]
        embedded_batch = np.array([embedding_matrix[indices] for indices in batch_sequences])
        batch_filename = os.path.join(output_dir, f'batch_{start // batch_size}.npy')
        np.save(batch_filename, embedded_batch)
        print(f'Saved {batch_filename}')

# 모든 배치를 로드하고 연결하는 함수
def load_and_concatenate_batches(output_dir='data/embedded_batches'):
    batch_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.npy')]
    batch_files.sort()  # 배치 파일들을 순서대로 로드
    concatenated_data = []
    for batch_file in batch_files:
        batch_data = np.load(batch_file)
        concatenated_data.append(batch_data)
    return np.concatenate(concatenated_data, axis=0)

# 클래스 분포를 출력하는 함수
def print_class_distribution(y, dataset_name):
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    print(f"{dataset_name}의 클래스 분포:")
    for label, count in distribution.items():
        print(f"클래스 {label}: {count}개 샘플, {count / len(y) * 100:.2f}%")

# 커스텀 어텐션 레이어 정의
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

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
    plt.savefig(f'img/BILSTM_{category}_curve.png')

if __name__ == '__main__':
    phishing_path = "data/augmented_transcriptions.xlsx"
    non_phishing_path = "data/combined_transcriptions.xlsx"  # 경로 수정

    phishing_df = pd.read_excel(phishing_path)
    non_phishing_df = pd.read_excel(non_phishing_path)

    phishing_df['label'] = 1
    non_phishing_df['label'] = 0

    # 피싱 및 비피싱 데이터를 결합
    combined_df = pd.concat([phishing_df, non_phishing_df], ignore_index=True)
    combined_df.dropna(subset=['transcript'], inplace=True)
    X = combined_df.drop(columns=['label'])
    y = combined_df['label']

    # 피싱 샘플의 수 계산
    phishing_count = y.value_counts()[1]

    # 비피싱 샘플의 수를 피싱 샘플 수의 3배로 설정
    desired_non_phishing_count = phishing_count * 3

    # 샘플링 전략 계산
    sampling_strategy = {0: desired_non_phishing_count, 1: phishing_count}

    # 랜덤 언더샘플링
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # 리샘플링된 특징과 타겟을 결합
    resampled_df = pd.concat([X_resampled, y_resampled], axis=1)

    # 토크나이저 사용하여 토큰화
    sentences = resampled_df['transcript'].tolist()
    labels = resampled_df['label'].tolist()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    max_length = max(len(sequence) for sequence in sequences)

    sequence_lengths = [len(seq) for seq in sequences]

    # 패딩 처리
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # 사전 학습된 Word2Vec 모델을 사용하여 임베딩
    word2vec_model = Word2Vec(sentences=sequences, vector_size=100, window=5, min_count=1, workers=4)
    embedding_matrix = word2vec_model.wv.vectors

    embedding_matrix = np.vstack([np.zeros((1, embedding_matrix.shape[1])), embedding_matrix])

    if np.max(padded_sequences) >= embedding_matrix.shape[0]:
        print("Error: Index out of range")
        print("Maximum index in padded sequences:", np.max(padded_sequences))

    max_index_padded_sequences = np.max(padded_sequences)
    max_index_embedding_matrix = embedding_matrix.shape[0] - 1  # Adjust for zero-based indexing

    if max_index_padded_sequences >= max_index_embedding_matrix:
        print("Error: Index out of range")
        print("Maximum index in padded sequences:", max_index_padded_sequences)
        print("Maximum index in embedding matrix:", max_index_embedding_matrix)

    embedded_sequences = np.array([embedding_matrix[indices] for indices in padded_sequences])

    batch_process_embedding(padded_sequences, embedding_matrix)
    embedded_sequences = load_and_concatenate_batches()
    print(embedded_sequences.shape)

    # 임베딩된 시퀀스를 디스크에 저장
    np.save('data/embedded_sequences.npy', embedded_sequences)

    file_path = 'data/embedded_sequences.npy'

    # 파일이 존재하는지 확인
    if os.path.exists(file_path):
        # 파일 삭제
        os.remove(file_path)
        print(f"File '{file_path}' deleted successfully.")
    else:
        print(f"File '{file_path}' does not exist.")

    # 레이블을 numpy 배열로 변환
    labels = np.array(labels)

    # 길이가 일치하는지 확인
    if len(labels) != len(embedded_sequences):
        print("Error: Length mismatch between labels and embedded sequences")
    else:
        # 인덱스를 섞기
        shuffled_indices = np.random.permutation(len(embedded_sequences))

        # 섞인 인덱스를 사용하여 임베딩된 시퀀스와 레이블을 섞기
        embedded_sequences_shuffled = embedded_sequences[shuffled_indices]
        labels_shuffled = labels[shuffled_indices]

        # 데이터를 훈련, 검증 및 테스트 세트로 분할 (계층화)
        X_train_temp, X_test, y_train_temp, y_test = train_test_split(
            embedded_sequences_shuffled, labels_shuffled, test_size=0.2, stratify=labels_shuffled, random_state=42
        )

        # 훈련 세트를 다시 훈련 및 검증 세트로 분할 (계층화)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_temp, y_train_temp, test_size=0.25, stratify=y_train_temp, random_state=42
        )

    # 클래스 분포 출력
    print_class_distribution(y_train, "훈련 세트")
    print_class_distribution(y_val, "검증 세트")
    print_class_distribution(y_test, "테스트 세트")

    # 모델 아키텍처 정의
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

    conv_layer = Conv1D(256, 5, activation='relu')(input_layer)
    conv_layer = BatchNormalization()(conv_layer)
    conv_layer = Dropout(0.5)(conv_layer)

    bidirectional_lstm_1 = Bidirectional(LSTM(128, return_sequences=True))(conv_layer)
    bidirectional_lstm_2 = Bidirectional(LSTM(64, return_sequences=True))(bidirectional_lstm_1)

    attention_layer = AttentionLayer()(bidirectional_lstm_2)

    global_max_pooling = GlobalMaxPooling1D()(bidirectional_lstm_2)
    dense_layer_1 = Dense(128, activation='relu')(global_max_pooling)
    dense_layer_1 = BatchNormalization()(dense_layer_1)
    dense_layer_1 = Dropout(0.5)(dense_layer_1)

    dense_layer_2 = Dense(64, activation='relu')(dense_layer_1)
    dense_layer_2 = BatchNormalization()(dense_layer_2)
    dense_layer_2 = Dropout(0.5)(dense_layer_2)

    output_layer = Dense(1, activation='sigmoid')(dense_layer_2)

    model = Model(inputs=input_layer, outputs=output_layer)

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    checkpoint = ModelCheckpoint('model/BILSTM_model.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_data=(X_val, y_val), callbacks=[early_stopping, checkpoint])

    # 에포크 번호 출력
    for epoch in range(1, len(history.history['loss']) + 1):
        print(f'Epoch {epoch}/{len(history.history["loss"])}')

    # 학습 및 검증 손실, 정확도, 정밀도, 재현율, F1 점수 저장
    train_accuracies = history.history['accuracy']
    val_accuracies = history.history['val_accuracy']
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    # 학습 및 검증 Precision, Recall, F1 Score 저장
    train_precisions = history.history['precision']
    val_precisions = history.history['val_precision']
    train_recalls = history.history['recall']
    val_recalls = history.history['val_recall']
    train_f1s = [2 * (p * r) / (p + r) for p, r in zip(train_precisions, train_recalls)]
    val_f1s = [2 * (p * r) / (p + r) for p, r in zip(val_precisions, val_recalls)]

    # 곡선 그리기
    plot('Loss')
    plot('Accuracy')
    plot('Precision')
    plot('Recall')
    plot('F1 Score')

    y_val_pred = model.predict(X_val)
    y_val_pred = (y_val_pred > 0.5).astype("int32")

    # 평가 지표 계산
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')