import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# 토큰 리스트를 노드로 하는 그래프 생성
def build_graph(X, y):
    """TF-IDF 벡터를 기반으로 그래프를 생성합니다."""
    G = nx.Graph()
    
    for i, vector in enumerate(X):
        print(i)
        for token_index in np.where(vector > 0)[0]:
            token = vectorizer.get_feature_names_out()[token_index]
            G.add_node(token)
            for j in np.where(vector > 0)[0]:
                if j != token_index:
                    neighbor_token = vectorizer.get_feature_names_out()[j]
                    G.add_edge(token, neighbor_token, weight=vector[token_index] * vector[j])
    return G

# GNN 모델 정의
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(data.num_node_features, 16)
        self.conv2 = GCNConv(16, 2)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

# 훈련 함수
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 평가 함수
def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data)
        loss = criterion(out[mask], data.y[mask]).item()
        _, pred = out[mask].max(dim=1)
        correct = pred.eq(data.y[mask]).sum().item()
        acc = correct / mask.sum().item()
        precision = precision_score(data.y[mask].cpu(), pred.cpu(), average='weighted')
        recall = recall_score(data.y[mask].cpu(), pred.cpu(), average='weighted')
        f1 = f1_score(data.y[mask].cpu(), pred.cpu(), average='weighted')
    return loss, acc, precision, recall, f1

# 곡선 plotting 함수
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
    plt.savefig(f'img/GNN_{category}_curve.png')


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

    # 'transcript' 열을 문자열로 변환
    combined_data['transcript'] = combined_data['transcript'].astype(str)

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(max_features=1000, tokenizer=lambda x: x.split())
    X = vectorizer.fit_transform(combined_data['transcript']).toarray()
    y = combined_data['label'].values

    # 데이터 확인
    print(X.shape)
    print(y.shape)

    # 그래프 생성
    G = build_graph(X, y)

    # 그래프 시각화
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=0.1)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()
    plt.savefig('img/graph_visualization.png')

    # 그래프 저장
    nx.write_gml(G, 'data/token_graph.gml')

    # 그래프 불러오기
    G = nx.read_gml('data/token_graph.gml')

    # 노드 이름을 인덱스로 변환
    node_list = list(G.nodes)
    node_to_index = {node: i for i, node in enumerate(node_list)}

    # 엣지를 숫자로 매핑하여 edge_index 생성
    edge_index = torch.tensor([(node_to_index[edge[0]], node_to_index[edge[1]]) for edge in G.edges], dtype=torch.long).t().contiguous()

    # PyTorch 텐서로 변환
    x = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.long)

    # 훈련, 검증, 테스트 마스크 설정
    num_train = int(0.6 * len(y))
    num_val = int(0.2 * len(y))
    train_mask = torch.zeros(len(y), dtype=torch.bool)
    val_mask = torch.zeros(len(y), dtype=torch.bool)
    test_mask = torch.zeros(len(y), dtype=torch.bool)
    train_mask[:num_train] = True
    val_mask[num_train:num_train+num_val] = True
    test_mask[num_train+num_val:] = True

    # 데이터 객체 생성
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # 모델, 손실 함수, 최적화기 정의
    model = GNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

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
    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train()
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(data.val_mask)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc, train_precision, train_recall, train_f1 = evaluate(data.train_mask)[1:]
        train_accuracies.append(train_acc)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)
        train_f1s.append(train_f1)
        val_accuracies.append(val_acc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train Accuracy: {train_acc}, Val Accuracy: {val_acc}, Train Precision: {train_precision}, Val Precision: {val_precision}, Train Recall: {train_recall}, Val Recall: {val_recall}, Train F1: {train_f1}, Val F1: {val_f1}')

    # 곡선 plotting
    plot('Loss')
    plot('Accuracy')
    plot('Precision')
    plot('Recall')
    plot('F1 Score')

    # 테스트 정확도 평가
    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(data.test_mask)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1 Score: {test_f1}')

    # 최종 모델 저장
    torch.save(model.state_dict(), 'model/gnn_model.pth')

    # 최종 결과 출력
    print(f'Final Training Loss: {train_losses[-1]}')
    print(f'Final Validation Loss: {val_losses[-1]}')
    print(f'Final Test Accuracy: {test_acc}')
    print(f'Final Test Precision: {test_precision}')
    print(f'Final Test Recall: {test_recall}')
    print(f'Final Test F1 Score: {test_f1}')




