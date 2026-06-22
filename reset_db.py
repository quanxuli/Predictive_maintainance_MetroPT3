from sqlalchemy import create_engine, text

# Kết nối tới Database
DB_CONNECTION_URL = "postgresql://postgres:anhquan26@localhost:5432/metro_pt3_db"
engine = create_engine(DB_CONNECTION_URL)

try:
    with engine.connect() as conn:
        # Xoá hoàn toàn bảng dữ liệu cũ
        conn.execute(text("DROP TABLE IF EXISTS sensor_realtime_predictions;"))
        conn.commit()
    print("✅ Đã xoá sạch dữ liệu cũ trong PostgreSQL!")
    print("👉 Bây giờ bạn có thể khởi động lại `api_server.py` và `realtime_predictor_lstm.py` để chạy lại từ đầu.")
except Exception as e:
    print(f"❌ Có lỗi xảy ra: {e}")
