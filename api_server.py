import uvicorn
from fastapi import FastAPI
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Metro-PT3 Realistic Replay API")

print("Đang tải dữ liệu gốc (Vui lòng đợi vài giây)...")
dataset_path = "MetroPT3.csv"
try:
    df = pd.read_csv(dataset_path)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp']) # Cần cột datetime để tìm kiếm
    df['timestamp'] = df['timestamp'].astype(str)
    df = df.ffill().bfill() 
    
    # --- TÌM VỊ TRÍ ĐỂ REPLAY (TRÁNH BIAS) ---
    # Trong train_offline_lstm.py, chúng ta đã chia 80% đầu làm Train, 20% cuối làm Test.
    # Sự kiện 18-04 nằm trong tập Train -> Mô hình đã "học thuộc" nó.
    # Để test khách quan (không Bias), chúng ta phải dùng sự kiện cuối cùng: '2020-07-15 14:30:00' (nằm trong tập Test).
    # Bắt đầu phát dữ liệu từ trước đó 20 phút (14:10:00)
    start_replay_time = pd.to_datetime('2020-07-15 14:10:00')
    
    mask = df['timestamp_dt'] >= start_replay_time
    if mask.any():
        start_index = mask.idxmax()
    else:
        start_index = 0
        
    print(f"Tải thành công! Bắt đầu tua nhanh đến dòng: {start_index} (Thời gian: {df.iloc[start_index]['timestamp']})")
    print(f"Lưu ý: Dữ liệu này nằm trong tập TEST, mô hình LSTM chưa từng nhìn thấy đoạn data này khi huấn luyện!")
    
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {dataset_path}.")
    df = pd.DataFrame()
    start_index = 0

current_index = start_index

@app.get("/")
def root():
    return {"message": "API Giả lập Cảm biến (Chế độ Replay Dữ liệu thật) đang hoạt động. Truy cập /stream để lấy dữ liệu."}

@app.get("/stream")
def get_sensor_stream(batch_size: int = 5):
    global current_index
    
    total_rows = len(df)
    if df.empty or total_rows == 0:
        return {"error": "Không có dữ liệu."}
        
    if current_index >= total_rows:
        current_index = start_index # Lặp lại từ đầu nếu hết data

    batch = df.iloc[current_index : current_index + batch_size].copy()
    current_index += batch_size
    
    # Xoá cột timestamp_dt phụ trợ
    batch = batch.drop(columns=['timestamp_dt'], errors='ignore')
    
    # In cảnh báo ra Terminal của API để đối chiếu với Predictor
    current_time = pd.to_datetime(batch.iloc[-1]['timestamp'])
    
    # Sự kiện Test Set: 2020-07-15 14:30:00
    if pd.to_datetime('2020-07-15 14:30:00') <= current_time <= pd.to_datetime('2020-07-15 19:00:00'):
        print(f"[{current_time}] ⚠️ [SỰ CỐ TEST SET] Đang phát lại dữ liệu lúc máy hỏng thật (Không có Bias)!")
    else:
        print(f"[{current_time}] 🟢 [Bình thường] Đang phát lại dữ liệu tốt.")

    return batch.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=True)
