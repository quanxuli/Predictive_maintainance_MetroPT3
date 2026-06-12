import requests
import time
import pandas as pd
from sqlalchemy import create_engine

# 1. Cấu hình API và Database
API_URL = "http://127.0.0.1:8000/stream?batch_size=5"

# Chuỗi kết nối PostgreSQL (Thay đổi username và password của bạn)
DB_CONNECTION_URL = "postgresql://postgres:anhquan26@localhost:5432/metro_pt3_db"

# Khởi tạo engine kết nối với SQLAlchemy
engine = create_engine(DB_CONNECTION_URL)

def consume_and_load_data():
    print("Bắt đầu hút dữ liệu cảm biến và Load vào PostgreSQL...")
    print("Nhấn Ctrl+C để dừng.\n" + "="*50)
    
    try:
        while True:
            # Gửi request lên API
            response = requests.get(API_URL)
            
            if response.status_code == 200:
                data = response.json()
                
                if not data:
                    print("Không có dữ liệu trả về từ API.")
                    time.sleep(2)
                    continue

                # Chuyển đổi JSON thành Pandas DataFrame
                df = pd.DataFrame(data)
                
                # Load dữ liệu vào bảng 'sensor_raw_data' trong PostgreSQL
                df.to_sql('sensor_raw_data', engine, if_exists='append', index=False)
                
                # In log ra màn hình Console cho chuẩn xác với bộ Metro-PT3
                print(f"[{time.strftime('%H:%M:%S')}] Đã insert {len(df)} dòng vào DB.")
                
                for row in data:
                    # Lấy đúng tên cột của dataset Metro-PT3
                    ts = row.get('timestamp')
                    tp2 = row.get('TP2')
                    oil_temp = row.get('Oil_temperature')
                    motor_current = row.get('Motor_current')
                    
                    # In ra một số thông số đại diện để theo dõi luồng
                    print(f"  -> Time: {ts} | Áp suất TP2: {tp2} | Nhiệt độ dầu: {oil_temp} | Dòng điện: {motor_current}")
                
                print("-" * 50)
            else:
                print(f"Lỗi kết nối API: HTTP {response.status_code}")
            
            # Tạm dừng 2 giây trước khi gọi lần tiếp theo
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nĐã dừng luồng hút và load dữ liệu an toàn.")
    except Exception as e:
        print(f"\nCó lỗi hệ thống xảy ra: {e}")

if __name__ == "__main__":
    consume_and_load_data()