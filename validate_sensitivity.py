import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

def run_validation_and_sensitivity():
    print("Bắt đầu Xác thực mô hình và Phân tích độ nhạy...")
    
    # 1. TẢI DỮ LIỆU VÀ MÔ HÌNH
    print("\n1. Đang tải dữ liệu và mô hình...")
    usecols = ['timestamp', 'TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
    df = pd.read_csv('MetroPT3.csv', usecols=usecols)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df = df.ffill().bfill()

    # Tính Rolling Features
    sensor_cols = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
    window_size = 10
    for col in sensor_cols:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=window_size, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=window_size, min_periods=1).std().fillna(0)

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

    feature_cols = [col for col in df.columns if 'rolling' in col]
    X = df[feature_cols]
    y = df['target']

    # Load Model
    model = joblib.load('rf_smote_model.pkl')

    # =========================================================
    # PHẦN 1: TIME-SERIES CROSS VALIDATION
    # =========================================================
    print("\n2. Thực hiện Time-Series Cross Validation (5 folds)...")
    tscv = TimeSeriesSplit(n_splits=5)
    fold = 1
    f1_scores = []
    
    for train_index, test_index in tscv.split(X):
        # Trích xuất tập test cho fold hiện tại
        X_test_fold, y_test_fold = X.iloc[test_index], y.iloc[test_index]
        
        # Bỏ qua fold nếu không có lỗi nào để test (tránh lỗi ZeroDivision)
        if y_test_fold.sum() == 0:
            continue
            
        y_pred_fold = model.predict(X_test_fold)
        f1 = f1_score(y_test_fold, y_pred_fold, zero_division=0)
        f1_scores.append(f1)
        print(f"   - Fold {fold}: F1-Score = {f1:.4f}")
        fold += 1
        
    if f1_scores:
        print(f"-> F1-Score trung bình (Cross-Validation): {np.mean(f1_scores):.4f}")

    # =========================================================
    # PHẦN 2: SENSITIVITY ANALYSIS (FEATURE IMPORTANCE)
    # =========================================================
    print("\n3. Phân tích độ nhạy của các biến (Feature Importance)...")
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance (%)': importances * 100
    }).sort_values(by='Importance (%)', ascending=False)
    
    print(feature_importance_df.head(5).to_string(index=False))

    # =========================================================
    # PHẦN 3: THRESHOLD SENSITIVITY ANALYSIS
    # =========================================================
    print("\n4. Phân tích độ nhạy của Ngưỡng cảnh báo (Threshold Sensitivity)...")
    # Lấy xác suất dự đoán lỗi trên toàn bộ tập dữ liệu
    y_probs = model.predict_proba(X)[:, 1]
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    print(f"{'Ngưỡng (Threshold)':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 55)
    
    for thresh in thresholds:
        # Điều chỉnh quyết định dựa trên ngưỡng
        y_pred_custom = (y_probs >= thresh).astype(int)
        
        prec = precision_score(y, y_pred_custom, zero_division=0)
        rec = recall_score(y, y_pred_custom, zero_division=0)
        f1 = f1_score(y, y_pred_custom, zero_division=0)
        
        print(f"Tỉ lệ >= {thresh:<14} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}")

if __name__ == "__main__":
    run_validation_and_sensitivity()