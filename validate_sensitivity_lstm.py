import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

class MetroLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1):
        super(MetroLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

def create_sequences(data, labels, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        xs.append(data[i:(i + seq_length)])
        ys.append(labels[i + seq_length - 1])
    return np.array(xs), np.array(ys)

def run_validation_and_sensitivity():
    print("Bắt đầu Xác thực mô hình LSTM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. TẢI DỮ LIỆU
    print("\n1. Đang tải dữ liệu...")
    usecols = ['timestamp', 'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
    df = pd.read_csv('MetroPT3.csv', usecols=usecols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.ffill().bfill()

    # Gán nhãn Ground Truth
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

    sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']

    # Tải Scaler và Model
    scaler = joblib.load('scaler.pkl')
    scaled_data = scaler.transform(df[sensor_cols])
    
    model = MetroLSTM(input_size=7, hidden_size=64, num_layers=1)
    model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
    model.to(device)
    model.eval()

    # Tạo sequences cho toàn bộ dữ liệu
    SEQ_LENGTH = 10
    X_seq, y_seq = create_sequences(scaled_data, df['target'].values, SEQ_LENGTH)

    # =========================================================
    # PHẦN 1: THRESHOLD SENSITIVITY ANALYSIS
    # =========================================================
    print("\n2. Phân tích độ nhạy của Ngưỡng cảnh báo (Threshold Sensitivity) bằng LSTM...")
    
    # Dự đoán xác suất trên toàn bộ dữ liệu theo từng batch nhỏ để tránh hết RAM GPU
    print("   Đang tính toán xác suất dự đoán (có thể mất vài phút)...")
    BATCH_SIZE = 2048
    y_probs = []
    
    with torch.no_grad():
        for i in range(0, len(X_seq), BATCH_SIZE):
            batch_x = torch.tensor(X_seq[i:i+BATCH_SIZE], dtype=torch.float32).to(device)
            outputs = model(batch_x)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            y_probs.extend(probs)
            
    y_probs = np.array(y_probs)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    print(f"\n{'Ngưỡng (Threshold)':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    
    for thresh in thresholds:
        y_pred_custom = (y_probs >= thresh).astype(int)
        
        prec = precision_score(y_seq, y_pred_custom, zero_division=0)
        rec = recall_score(y_seq, y_pred_custom, zero_division=0)
        f1 = f1_score(y_seq, y_pred_custom, zero_division=0)
        
        print(f"Tỉ lệ >= {thresh:<14} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

    print("\nLưu ý: LSTM là mô hình Black-box (Hộp đen), nên việc trích xuất Feature Importance không áp dụng trực tiếp như Random Forest.")

if __name__ == "__main__":
    run_validation_and_sensitivity()
