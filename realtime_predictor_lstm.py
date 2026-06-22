import time
import requests
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
from sqlalchemy import create_engine
import warnings

warnings.filterwarnings('ignore')

# 1. CẤU HÌNH HỆ THỐNG
API_URL = "http://127.0.0.1:8000/stream?batch_size=5"
DB_CONNECTION_URL = "postgresql://postgres:anhquan26@localhost:5432/metro_pt3_db"
engine = create_engine(DB_CONNECTION_URL)

# Khai báo lại kiến trúc LSTM
class MetroLSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, num_layers=1):
        super(MetroLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

print("Đang tải mô hình LSTM và Scaler...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetroLSTM(input_size=7, hidden_size=64, num_layers=1)
model.load_state_dict(torch.load('lstm_model.pth', map_location=device))
model.to(device)
model.eval()

scaler = joblib.load('scaler.pkl')

sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
window_size = 10
buffer_df = pd.DataFrame()

def run_realtime_prediction():
    global buffer_df
    print("Bắt đầu lắng nghe luồng dữ liệu cảm biến (LSTM)...")
    print("="*50)
    
    try:
        while True:
            response = requests.get(API_URL)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    time.sleep(2)
                    continue

                new_df = pd.DataFrame(data)
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

                buffer_df = pd.concat([buffer_df, new_df], ignore_index=True)

                # Giữ lại tối đa 20 dòng gần nhất để tạo sequence (Window=10)
                if len(buffer_df) > 20:
                    buffer_df = buffer_df.tail(20).reset_index(drop=True)

                    # Cần ít nhất window_size (10) dòng để bắt đầu dự đoán
                if len(buffer_df) >= window_size:
                    # Chuẩn hóa dữ liệu trong buffer
                    scaled_buffer = scaler.transform(buffer_df[sensor_cols])
                    
                    # Tạo sequences cho các dòng dữ liệu MỚI NHẤT (new_df)
                    sequences = []
                    valid_indices = [] # Theo dõi dòng nào đủ điều kiện dự đoán
                    num_new_rows = len(new_df)
                    total_rows = len(scaled_buffer)
                    
                    for i in range(total_rows - num_new_rows, total_rows):
                        # Chỉ lấy sequence nếu lịch sử đủ 10 dòng
                        if i >= window_size - 1:
                            seq = scaled_buffer[i - window_size + 1 : i + 1]
                            sequences.append(seq)
                            valid_indices.append(i - (total_rows - num_new_rows))
                            
                    # Mặc định dự đoán là 0 (Bình thường)
                    predictions = np.zeros(num_new_rows, dtype=int)
                    
                    if len(sequences) > 0:
                        sequences = np.array(sequences)
                        X_live = torch.tensor(sequences, dtype=torch.float32).to(device)

                        # CHẠY MÔ HÌNH DỰ ĐOÁN
                        with torch.no_grad():
                            outputs = model(X_live)
                            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                            preds = (probs >= 0.5).astype(int)
                            
                        # Gán kết quả vào đúng vị trí
                        for idx, p in zip(valid_indices, preds):
                            predictions[idx] = p

                    # Gắn nhãn và lưu vào Database
                    results_df = new_df.copy()
                    results_df['target'] = predictions
                    results_df['status'] = results_df['target'].apply(lambda x: 'Anomaly' if x == 1 else 'Normal')
                    
                    # Thêm Anomaly Score (dạng phần trăm) vào Database để hiển thị lên GUI
                    anomaly_scores = np.zeros(num_new_rows, dtype=float)
                    for idx, prob in zip(valid_indices, probs):
                        anomaly_scores[idx] = prob * 100
                    results_df['anomaly_score'] = anomaly_scores
                    
                    results_df.to_sql('sensor_realtime_predictions', engine, if_exists='append', index=False)

                    # HIỂN THỊ CẢNH BÁO (KÈM XÁC SUẤT)
                    anomalies = results_df[results_df['status'] == 'Anomaly']
                    avg_prob = np.mean(probs) * 100 # Tính xác suất trung bình của đợt dữ liệu này
                    
                    if not anomalies.empty:
                        print(f"[{time.strftime('%H:%M:%S')}] 🔴 CẢNH BÁO MỨC ĐỎ: Phát hiện {len(anomalies)} dấu hiệu rò rỉ khí! (Độ tin cậy: {avg_prob:.1f}%)")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] 🟢 Đã quét {len(new_df)} dòng -> Bình thường (Khả năng lỗi: {avg_prob:.2f}%)")

            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nĐã tắt hệ thống dự đoán.")
    except Exception as e:
        print(f"Lỗi luồng: {e}")

if __name__ == "__main__":
    run_realtime_prediction()
