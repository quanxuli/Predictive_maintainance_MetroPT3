from fastapi import FastAPI
import pandas as pd

app = FastAPI(title="Metro-PT3 Sensor Stream API")

# 1. Tải dữ liệu vào bộ nhớ (Thực tế file này khá nặng ~1.5 triệu dòng)
print("Đang tải dữ liệu, vui lòng đợi...")
dataset_path = "MetroPT3.csv"
try:
    df = pd.read_csv(dataset_path)
    print(f"Tải thành công! Tổng số dòng: {len(df)}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {dataset_path}. Vui lòng kiểm tra lại.")
    df = pd.DataFrame()

# Biến toàn cục để theo dõi vị trí dòng dữ liệu đang đọc
current_index = 0

@app.get("/")
def root():
    return {"message": "API Giả lập Cảm biến Metro-PT3 đang hoạt động. Truy cập /stream để lấy dữ liệu."}

@app.get("/stream")
def get_sensor_data(batch_size: int = 5):
    """
    Endpoint này trả về 'batch_size' dòng dữ liệu mỗi lần gọi.
    Mặc định trả về 5 dòng/lần.
    """
    global current_index
    
    if df.empty:
        return {"error": "Dataset chưa được tải."}

    # Tính toán vị trí kết thúc của batch hiện tại
    end_index = current_index + batch_size
    
    # Lấy dữ liệu
    if end_index >= len(df):
        # Nếu đã đọc đến cuối file, lấy phần còn lại và reset index về 0 để lặp lại
        chunk = df.iloc[current_index : len(df)]
        current_index = 0
    else:
        # Lấy dữ liệu theo batch
        chunk = df.iloc[current_index : end_index]
        current_index = end_index

    # Trả về định dạng JSON (danh sách các dictionary)
    return chunk.to_dict(orient="records")  