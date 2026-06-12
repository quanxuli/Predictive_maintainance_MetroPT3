import time
import requests
import pandas as pd
import joblib
from sqlalchemy import create_engine
import warnings

# Tắt cảnh báo của Pandas cho màn hình Terminal gọn gàng
warnings.filterwarnings('ignore')

# 1. CẤU HÌNH HỆ THỐNG
API_URL = "http://127.0.0.1:8000/stream?batch_size=5"
DB_CONNECTION_URL = "postgresql://postgres:anhquan26@localhost:5432/metro_pt3_db"
engine = create_engine(DB_CONNECTION_URL)

# Tải "Bộ não" AI đã được huấn luyện offline
print("Đang tải mô hình AI vào bộ nhớ RAM...")
model = joblib.load('rf_smote_model.pkl')

sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
window_size = 10

# Khởi tạo Memory Buffer để lưu tạm dữ liệu tính toán (Không lưu vào ổ cứng)
buffer_df = pd.DataFrame()

def run_realtime_prediction():
    global buffer_df
    print("Bắt đầu lắng nghe luồng dữ liệu cảm biến...")
    print("="*50)
    
    try:
        while True:
            # 1. Hút dữ liệu từ API
            response = requests.get(API_URL)
            if response.status_code == 200:
                data = response.json()
                if not data:
                    time.sleep(2)
                    continue

                new_df = pd.DataFrame(data)
                new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])

                # 2. Đưa dữ liệu mới vào Bộ đệm
                buffer_df = pd.concat([buffer_df, new_df], ignore_index=True)

                # Giữ lại tối đa 20 dòng gần nhất để tối ưu RAM
                if len(buffer_df) > 20:
                    buffer_df = buffer_df.tail(20).reset_index(drop=True)

                # 3. Tính toán Đặc trưng (Feature Engineering) trên luồng
                if len(buffer_df) >= window_size:
                    temp_df = buffer_df.copy()
                    
                    # Tính Rolling Mean & Std giống hệt pha Offline
                    for col in sensor_cols:
                        temp_df[f'{col}_rolling_mean'] = temp_df[col].rolling(window=window_size, min_periods=1).mean()
                        temp_df[f'{col}_rolling_std'] = temp_df[col].rolling(window=window_size, min_periods=1).std().fillna(0)

                    # Chỉ lấy đúng các dòng mới nhất (vừa tải về) để dự đoán
                    features_to_predict = temp_df.tail(len(new_df))
                    feature_cols = [col for col in features_to_predict.columns if 'rolling' in col]
                    X_live = features_to_predict[feature_cols].fillna(0)

                    # 4. CHẠY MÔ HÌNH DỰ ĐOÁN
                    predictions = model.predict(X_live)

                    # 5. Gắn nhãn và lưu vào Database cho Dashboard
                    results_df = new_df.copy() # Chỉ lưu data gốc kèm theo nhãn cảnh báo
                    results_df['target'] = predictions
                    results_df['status'] = results_df['target'].apply(lambda x: 'Anomaly' if x == 1 else 'Normal')

                    # Lưu vào bảng chuyên dụng cho Dashboard Realtime
                    results_df.to_sql('sensor_realtime_predictions', engine, if_exists='append', index=False)

                    # 6. HIỂN THỊ CẢNH BÁO LÊN TERMINAL
                    anomalies = results_df[results_df['status'] == 'Anomaly']
                    if not anomalies.empty:
                        print(f"[{time.strftime('%H:%M:%S')}] 🔴 CẢNH BÁO MỨC ĐỎ: Phát hiện {len(anomalies)} dấu hiệu rò rỉ khí!")
                    else:
                        print(f"[{time.strftime('%H:%M:%S')}] 🟢 Đã quét {len(new_df)} dòng -> Hệ thống vận hành Bình thường.")

            time.sleep(2) # Đợi 2 giây cho nhịp API tiếp theo
            
    except KeyboardInterrupt:
        print("\nĐã tắt hệ thống dự đoán.")
    except Exception as e:
        print(f"Lỗi luồng: {e}")

if __name__ == "__main__":
    run_realtime_prediction()