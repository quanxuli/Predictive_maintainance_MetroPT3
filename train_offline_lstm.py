"""
Mô hình Predictive Maintenance sử dụng Deep Learning (LSTM) bằng PyTorch
"""
__author__ = "ZEPHYRUS"

import pandas as pd
import numpy as np
import time
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# 1. Định nghĩa Kiến trúc LSTM
class MetroLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1):
        super(MetroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Lớp LSTM (batch_first=True vì input là [batch, seq_len, features])
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Lớp Linear (Fully Connected) để output ra 1 giá trị (xác suất lỗi)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # h0 và c0 tự động khởi tạo bằng 0 nếu không truyền vào
        out, (hn, cn) = self.lstm(x)
        # Lấy output của bước thời gian cuối cùng trong sequence
        out = self.fc(out[:, -1, :]) 
        return out

def create_sequences(data, labels, seq_length):
    xs = []
    ys = []
    # Khởi tạo ma trận rỗng để tối ưu tốc độ
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(labels[i + seq_length - 1]) # Lấy nhãn của điểm cuối cùng trong window
    return np.array(xs), np.array(ys)

def build_offline_model_lstm():
    print("Bắt đầu quy trình Offline Training với PyTorch LSTM...")
    start_time = time.time()
    
    # Kiểm tra GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị tính toán: {device}")

    try:
        # 1. ĐỌC DỮ LIỆU
        print("1. Đang tải tập dữ liệu gốc...")
        usecols = ['timestamp', 'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        df = pd.read_csv('MetroPT3.csv', usecols=usecols)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.ffill().bfill()

        # 2. GẮN NHÃN (Ground Truth)
        print("2. Đang gán nhãn sự cố...")
        df['target'] = 0 
        failure_windows = [
            ('2020-04-18 00:00:00', '2020-04-18 23:59:59'),
            ('2020-05-29 23:30:00', '2020-05-30 06:00:00'),
            ('2020-06-05 10:00:00', '2020-06-07 14:30:00'),
            ('2020-07-15 14:30:00', '2020-07-15 19:00:00')
        ]
        
        for start, end in failure_windows:
            mask = (df['timestamp'] >= start) & (df['timestamp'] <= end)
            df.loc[mask, 'target'] = 1

        # 3. CHIA TẬP TRAIN/TEST THEO THỜI GIAN
        print("3. Đang chia tập Train/Test và Chuẩn hóa (Scaling)...")
        sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        
        # Để tránh data leakage, ta chia Train/Test trước khi scale
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Scaling
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[sensor_cols])
        test_scaled = scaler.transform(test_df[sensor_cols])
        
        # Lưu scaler
        joblib.dump(scaler, 'scaler.pkl')
        print("-> Đã lưu StandardScaler vào 'scaler.pkl'")

        # 4. TẠO SEQUENCE CHO LSTM
        print("4. Đang tạo các Sequences (Cửa sổ trượt = 10)...")
        SEQ_LENGTH = 10
        X_train_seq, y_train_seq = create_sequences(train_scaled, train_df['target'].values, SEQ_LENGTH)
        X_test_seq, y_test_seq = create_sequences(test_scaled, test_df['target'].values, SEQ_LENGTH)
        
        print(f"-> Dữ liệu Train (Sequence): {X_train_seq.shape}. Số điểm lỗi: {y_train_seq.sum()}")
        
        # Tính trọng số phạt cho lớp lỗi (do mất cân bằng) thay vì dùng SMOTE
        num_neg = (y_train_seq == 0).sum()
        num_pos = (y_train_seq == 1).sum()
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)], dtype=torch.float32).to(device)

        # 5. CHUẨN BỊ DATALOADER
        BATCH_SIZE = 1024 # Batch lớn để train nhanh trên GPU
        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32), 
                                      torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1))
        test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32), 
                                     torch.tensor(y_test_seq, dtype=torch.float32).unsqueeze(1))

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 6. KHỞI TẠO MÔ HÌNH
        model = MetroLSTM(input_size=7, hidden_size=64, num_layers=1).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 7. HUẤN LUYỆN
        EPOCHS = 5
        print(f"\n5. Bắt đầu huấn luyện mô hình ({EPOCHS} Epochs)...")
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
                
            train_loss /= len(train_loader.dataset)
            print(f"   Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}")

        # 8. ĐÁNH GIÁ TRÊN TẬP TEST
        print("\n--- ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ (TEST SET) ---")
        model.eval()
        y_preds = []
        y_trues = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                # Kích hoạt sigmoid và áp dụng ngưỡng 0.5
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int)
                
                y_preds.extend(preds)
                y_trues.extend(y_batch.numpy())

        y_trues = np.array(y_trues)
        y_preds = np.array(y_preds)

        print(classification_report(y_trues, y_preds, labels=[0, 1], target_names=['Bình thường (0)', 'Lỗi (1)'], zero_division=0))
        
        conf_matrix = confusion_matrix(y_trues, y_preds, labels=[0, 1])
        print("Ma trận nhầm lẫn:")
        print(f"  - Bình thường đoán đúng: {conf_matrix[0][0]}")
        print(f"  - Lỗi đoán trúng (TP): {conf_matrix[1][1]}")
        print(f"  - Báo động giả (FP): {conf_matrix[0][1]}")
        print(f"  - Bỏ sót lỗi (FN): {conf_matrix[1][0]}")

        # 9. LƯU MÔ HÌNH
        print("\n6. Đang lưu trọng số mô hình...")
        torch.save(model.state_dict(), 'lstm_model.pth')
        print(f"-> Xuất file thành công: 'lstm_model.pth'.")
        print(f"Tổng thời gian chạy: {round(time.time() - start_time, 2)} giây.")

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    build_offline_model_lstm()
