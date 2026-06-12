import uvicorn
from fastapi import FastAPI
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Metro-PT3 Sensor Stream API")

print("Đang tải dữ liệu, vui lòng đợi...")
try:
    df = pd.read_csv("MetroPT3.csv")
    
    # Ép kiểu datetime để dễ tìm kiếm chính xác
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.ffill().bfill()
    total_rows = len(df)
    
    # -------------------------------------------------------------
    # TUA NHANH ĐẾN ĐÚNG 1 PHÚT TRƯỚC KHI MÁY HỎNG THẬT
    # Sự cố thật bắt đầu vào: 2020-04-18 00:00:00
    # Ta bắt đầu phát từ:      2020-04-17 23:59:00
    # -------------------------------------------------------------
    error_time = pd.to_datetime('2020-04-17 23:59:00')
    start_idx = df[df['timestamp'] >= error_time].index[0]
    current_index = start_idx
    
    # Ép lại thành string cho API trả về JSON
    df['timestamp'] = df['timestamp'].astype(str)
    
    print(f"-> Tải thành công {total_rows} dòng.")
    print(f"-> ⏳ CỖ MÁY THỜI GIAN: Đã tua đến dòng {current_index} (Ngay trước khi hỏng).")

except Exception as e:
    print(f"Lỗi hệ thống: {e}")
    df = pd.DataFrame()
    total_rows = 0
    current_index = 0

@app.get("/")
def root():
    return {"message": "API Streaming đang hoạt động."}

@app.get("/stream")
def get_sensor_stream(batch_size: int = 5):
    global current_index
    
    if df.empty or total_rows == 0:
        return {"error": "Không có dữ liệu."}
        
    if current_index >= total_rows:
        current_index = 0

    # Lấy dữ liệu THỰC TẾ 100% từ lịch sử
    batch = df.iloc[current_index : current_index + batch_size].copy()
    current_index += batch_size
    
    return batch.to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")