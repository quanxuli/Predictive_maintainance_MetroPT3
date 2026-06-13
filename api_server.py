import uvicorn
from fastapi import FastAPI
import pandas as pd
import random
import warnings

warnings.filterwarnings('ignore')

app = FastAPI(title="Metro-PT3 Sensor Stream API")
# uvicorn api_server:app 
# 1. Tải dữ liệu vào bộ nhớ
print("Đang tải dữ liệu, vui lòng đợi...")
dataset_path = "MetroPT3.csv"
try:
    df = pd.read_csv(dataset_path)
    
    # Ép kiểu timestamp về chuỗi và lấp đầy các ô trống để API không bị lỗi JSON
    df['timestamp'] = df['timestamp'].astype(str)
    df = df.ffill().bfill() 
    
    total_rows = len(df) # <-- FIX LỖI Ở ĐÂY: Đã định nghĩa total_rows
    print(f"Tải thành công! Tổng số dòng: {total_rows}")
    
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {dataset_path}. Vui lòng kiểm tra lại.")
    df = pd.DataFrame()
    total_rows = 0 # Đảm bảo total_rows tồn tại ngay cả khi lỗi đọc file

# Biến toàn cục để theo dõi vị trí dòng dữ liệu đang đọc
current_index = 0
is_leaking = False      # Trạng thái máy: Có đang rò rỉ không?
leak_severity = 0.0     # Mức độ rò rỉ (từ 0.0 đến 1.0+)

@app.get("/")
def root():
    return {"message": "API Giả lập Cảm biến Metro-PT3 đang hoạt động. Truy cập /stream để lấy dữ liệu."}

@app.get("/stream")
def get_sensor_stream(batch_size: int = 5):
    global current_index, is_leaking, leak_severity
    
    if df.empty or total_rows == 0:
        return {"error": "Không có dữ liệu."}
        
    if current_index >= total_rows:
        current_index = 0

    batch = df.iloc[current_index : current_index + batch_size].copy()
    current_index += batch_size
    
    # -------------------------------------------------------------
    # MÔ PHỎNG QUÁ TRÌNH SUY THOÁI TỰ NHIÊN (GRADUAL DEGRADATION)
    # -------------------------------------------------------------
    
    # Xác suất 2% bắt đầu xuất hiện vết nứt nhỏ (Chỉ kích hoạt nếu đang bình thường)
    if not is_leaking and random.random() < 0.02:
        is_leaking = True
        leak_severity = 0.0
        print("\n[CƠ HỌC] ⚠️ Bắt đầu xuất hiện vết nứt ống khí. Khí đang rò rỉ chậm...")

    # Nếu đang trong quá trình rò rỉ
    if is_leaking:
        # Tăng tốc độ hư hỏng (mỗi nhịp tồi tệ thêm 10% thay vì 5%)
        leak_severity += 0.10 
        
        # 1. Ép áp suất tụt mạnh hơn (Tụt thẳng về 0 khi hỏng nặng)
        batch['TP2'] = batch['TP2'] * (1 - leak_severity)
        
        # 2. Dòng điện và Nhiệt độ phải tăng vọt kịch trần
        batch['Motor_current'] = batch['Motor_current'] + (leak_severity * 10.0)
        batch['Oil_temperature'] = batch['Oil_temperature'] + (leak_severity * 20.0)
        
        # 3. Ép Van xả đóng hoàn toàn (Triệu chứng rõ nhất của rò khí)
        batch['DV_pressure'] = 0.0 

        # 4. ĐẶC BIỆT QUAN TRỌNG: Tạo độ rung lắc hỗn loạn cho H1 (Feature quan trọng thứ 2)
        import numpy as np
        batch['H1'] = batch['H1'] * np.random.uniform(0.2, 3.0, size=len(batch))

        # Nếu hỏng quá nặng (vượt mức 1.0)
        if leak_severity > 1.0:
            print("[CƠ HỌC] 🔧 Kỹ sư đã can thiệp sửa chữa. Reset hệ thống về bình thường.\n")
            is_leaking = False
            leak_severity = 0.0
    # -------------------------------------------------------------

    return batch.to_dict(orient="records")
