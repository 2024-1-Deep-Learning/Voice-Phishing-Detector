import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import os

# 사용자 정의 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
# 모델 클래스 정의
class ClassificationModel(nn.Module):
    def __init__(self):
        super(ClassificationModel, self).__init__()
        self.den1 = nn.Linear(768, 1024)
        self.den2 = nn.Linear(1024, 512)
        self.den3 = nn.Linear(512, 256)
        self.den4 = nn.Linear(256, 128)
        self.den5 = nn.Linear(128, 64)
        self.den6 = nn.Linear(64, 32)
        self.den7 = nn.Linear(32, 8)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.batch_norm5 = nn.BatchNorm1d(64)
        self.batch_norm6 = nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.den1(x)))
        x = F.relu(self.batch_norm2(self.den2(x)))
        x = F.relu(self.batch_norm3(self.den3(x)))
        x = F.relu(self.batch_norm4(self.den4(x)))
        x = F.relu(self.batch_norm5(self.den5(x)))
        x = F.relu(self.batch_norm6(self.den6(x)))
        x = self.dropout(x)
        x = self.den7(x)
        return x
    
# 평가 함수
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output = model(batch_x)
            loss = criterion(output, batch_y)
            total_loss += loss.item()
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    return avg_loss, accuracy, precision, recall, f1

# 곡선 그리는 함수
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
    plt.plot(range(1, num_epochs + 1), train, label=f'Training {category}')
    plt.plot(range(1, num_epochs + 1), val, label=f'Validation {category}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{category}')
    plt.title(f'{category} Curve')
    plt.legend()
    plt.savefig(f'img/koSBERT_{category}_curve.png')

if __name__ == '__main__':
    # 로컬 경로에서 KoSimCSE 모델과 토크나이저 불러오기
    model_path = './KoSentenceBERT_SKTBERT/KoSimCSE-roberta-multitask'
    model = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'
    tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')  # or 'BM-K/KoSimCSE-bert-multitask'

    # 데이터 로드
    phishing_path = "data/augmented_transcriptions.xlsx"
    non_phishing_path = "data/combined_transcriptions.xlsx"  # 경로 수정

    phishing_df = pd.read_excel(phishing_path)
    non_phishing_df = pd.read_excel(non_phishing_path)

    phishing_df['label'] = 1
    non_phishing_df['label'] = 0

    # 피싱 및 비피싱 데이터를 결합
    combined_df = pd.concat([phishing_df, non_phishing_df], ignore_index=True)
    combined_df.dropna(subset=['transcript'], inplace=True)

    # 문자열이 아닌 데이터 확인
    non_str_indices = combined_df[~combined_df['transcript'].apply(lambda x: isinstance(x, str))].index
    if len(non_str_indices) > 0:
        print(f"Non-string data found at indices: {non_str_indices.tolist()}")

    # 문자열로 변환
    combined_df['transcript'] = combined_df['transcript'].astype(str)

    # 임베딩 벡터 추출
    sent_embs = []
    for idx in tqdm(range(combined_df.shape[0])):
        sentence = combined_df.loc[idx, 'transcript']
        inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs, return_dict=False)[0]
        sent_emb = embeddings.mean(axis=1)
        sent_emb_np = sent_emb.detach().numpy()
        sent_embs.append(sent_emb_np)

    # numpy 배열로 변환
    sent_embs = np.vstack(sent_embs)

    # 임베딩 데이터를 DataFrame으로 변환
    emd_df = pd.DataFrame(sent_embs)

    # CSV 파일로 저장
    emd_df.to_csv('data/koSBERT_embs.csv', index=False, header=False)
    emd_df = pd.read_csv('data/koSBERT_embs.csv', header=None)

    # 라벨 데이터 추가
    emd_df = emd_df.assign(label=combined_df['label'])

    X = emd_df.drop(['label'], axis=1)
    y = emd_df['label']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    print(X_train.shape)
    print(X_test.shape)
    print(len(y_train))
    print(len(y_test))

    # 훈련 데이터셋과 검증 데이터셋으로 추가 분리
    X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # 클래스 분포 출력
    num_samp_cls = y_train.value_counts().values
    print(y_train.value_counts())
    print(y_test.value_counts())

    # 데이터 정규화
    minmax = MinMaxScaler()
    X_scaled_trn = minmax.fit_transform(X_train)
    X_scaled_val = minmax.transform(X_val)
    X_scaled_tes = minmax.transform(X_test)

    y_train_val = y_train_val.values
    y_val = y_val.values
    y_test = y_test.values

    # 데이터셋 생성
    train_data = CustomDataset(X_scaled_trn, y_train)
    test_data = CustomDataset(X_scaled_tes, y_test)
    val_data = CustomDataset(X_scaled_val, y_val)

    # 데이터 로더 생성
    train_loader = DataLoader(train_data, batch_size=400, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=400, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=400, shuffle=True)

    # 모델, 옵티마이저, 손실 함수 정의
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClassificationModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 손실과 정확도 저장용 리스트
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []

    # 훈련 루프
    for epoch in range(100):
        model.train()
        total_trn_loss = 0.0
        all_preds = []
        all_labels = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            output = model(batch_x)
            batch_loss = criterion(output, batch_y)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_trn_loss += batch_loss.item()
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

        avg_trn_loss = total_trn_loss / len(train_loader.dataset)
        train_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        train_precision = precision_score(all_labels, all_preds, average='weighted')
        train_recall = recall_score(all_labels, all_preds, average='weighted')
        train_f1 = f1_score(all_labels, all_preds, average='weighted')

        train_losses.append(avg_trn_loss)
        train_accuracies.append(train_accuracy)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)

        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        print(f'Epoch {epoch+1}, Train Loss: {avg_trn_loss}, Val Loss: {val_loss}, '
              f'Train Accuracy: {train_accuracy}, Val Accuracy: {val_accuracy}, '
              f'Train Precision: {train_precision}, Val Precision: {val_precision}, '
              f'Train Recall: {train_recall}, Val Recall: {val_recall}, '
              f'Train F1: {train_f1}, Val F1: {val_f1}')

        # 모델 저장
        torch.save(model.state_dict(), './model.pth')
        
    num_epochs = len(train_losses)

    # 곡선 그리기
    plot('Loss')
    plot('Accuracy')
    plot('Precision')
    plot('Recall')
    plot('F1 Score')

    # 테스트 정확도 평가
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(test_loader)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1 Score: {test_f1}')

    # 최종 모델 저장
    torch.save(model.state_dict(), 'final_model.pth')

    # 최종 결과 출력
    print(f'Final Training Loss: {train_losses[-1]}')
    print(f'Final Validation Loss: {val_losses[-1]}')
    print(f'Final Test Accuracy: {test_acc}')
    print(f'Final Test Precision: {test_precision}')
    print(f'Final Test Recall: {test_recall}')
    print(f'Final Test F1 Score: {test_f1}')
