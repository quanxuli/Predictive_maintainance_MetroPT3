"""
Mô hình Predictive Maintenance sử dụng Random Forest và SMOTE
"""
__author__ = "ZEPHYRUS"

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import time

def build_offline_model_smote():
    print("Bắt đầu quy trình Offline Training với SMOTE...")
    start_time = time.time()
    
    try:
        # 1. ĐỌC VÀ LÀM SẠCH DỮ LIỆU
        print("1. Đang tải tập dữ liệu gốc...")
        usecols = ['timestamp', 'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        df = pd.read_csv('MetroPT3.csv', usecols=usecols)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df = df.ffill().bfill()

        # 2. FEATURE ENGINEERING
        print("2. Đang tính toán các đặc trưng (Rolling)...")
        sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
        window_size = 10
        
        for col in sensor_cols:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)

        # 3. GẮN NHÃN (Ground Truth)
        print("3. Đang gán nhãn sự cố...")
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

        # 4. CHIA TẬP DỮ LIỆU
        feature_cols = [col for col in df.columns if 'rolling' in col]
        X = df[feature_cols]
        y = df['target']

        # Chia Train/Test theo thời gian
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        print(f"-> Dữ liệu Train gốc: {len(y_train)} dòng. Số điểm lỗi: {y_train.sum()}")

        # 5. ÁP DỤNG SMOTE (Chỉ trên tập Train)
        print("4. Đang sinh dữ liệu nhân tạo bằng SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        print(f"-> Dữ liệu Train sau SMOTE: {len(y_train_smote)} dòng. Số điểm lỗi: {y_train_smote.sum()}")

        # 6. HUẤN LUYỆN RANDOM FOREST
        print("5. Bắt đầu huấn luyện mô hình...")
        # Vì SMOTE đã cân bằng số lượng 50/50, ta không cần dùng class_weight='balanced' nữa
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_train_smote, y_train_smote)

        # 7. ĐÁNH GIÁ TRÊN TẬP TEST GỐC
        print("\n--- ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ (TEST SET) ---")
        y_pred = model.predict(X_test)
        
        # Thêm labels=[0, 1] và zero_division=0 để tránh lỗi khi thiếu class
        print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['Bình thường (0)', 'Lỗi (1)'], zero_division=0))
        
        # Thêm labels=[0, 1] để ma trận luôn cố định kích thước 2x2
        conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        print("Ma trận nhầm lẫn:")
        print(f"  - Bình thường đoán đúng: {conf_matrix[0][0]}")
        print(f"  - Lỗi đoán trúng (TP): {conf_matrix[1][1]}")
        print(f"  - Báo động giả (FP): {conf_matrix[0][1]}")
        print(f"  - Bỏ sót lỗi (FN): {conf_matrix[1][0]}")

        # 8. XUẤT FILE MODEL
        print("\n6. Đang lưu mô hình...")
        joblib.dump(model, 'rf_smote_model.pkl')
        print(f"-> Xuất file thành công: 'rf_smote_model.pkl'.")
        print(f"Tổng thời gian chạy: {round(time.time() - start_time, 2)} giây.")

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")

if __name__ == "__main__":
    build_offline_model_smote()