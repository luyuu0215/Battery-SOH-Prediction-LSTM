import os
import sys
from datetime import datetime
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Matplotlib 中文顯示配置
# ----------------------------
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------------
# 超參數與實驗設定
# ----------------------------
DROPOUT_LIST = [0.25, 0.3, 0.35]

N_RUNS = 15
MAX_EPOCHS = 500
BATCH_SIZE = 32
SEQ_LEN = 40

LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
LSTM_UNITS_3 = 32

EARLY_STOP_PATIENCE = 80

# 切分比例: 60% 訓練 / 20% 驗證 / 20% 測試
TEST_RATIO = 0.2
VAL_RATIO = 0.25

LEARNING_RATE = 0.003
RANDOM_STATE = 42

# ----------------------------
# 裝置設定
# ----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----------------------------
# 路徑與檔案設定
# ----------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_FILE = r"C:\Users\Andy\Desktop\battery_features_final_linear_soh_B0005~B0018.csv"

TEST_NAME = f"BiLSTM_PyTorch_{DROPOUT_LIST}_{SEQ_LEN}"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 特徵與標籤設定
# ----------------------------
FEATURES = [
    'IC_Peak_Height', 'IC_Peak_Voltage', 'IC_Valley_1_Magnitude',
    'CA_Area', 'CC_Charging_Time', 'CV_Charging_Time'
]
TARGET = 'SOH'


# ----------------------------
# Dataset
# ----------------------------
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------------
# PyTorch 模型 (修改為 BiLSTM)
# 注意：nn.LSTM 的內部 activation 為 tanh，無法直接改成 relu
# ----------------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, hidden3, dropout):
        super().__init__()

        # 加入 bidirectional=True
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden1,
            batch_first=True,
            bidirectional=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # 上一層為雙向，因此輸入維度要乘以 2
        self.lstm2 = nn.LSTM(
            input_size=hidden1 * 2,
            hidden_size=hidden2,
            batch_first=True,
            bidirectional=True
        )
        self.dropout2 = nn.Dropout(dropout)

        # 上一層為雙向，因此輸入維度要乘以 2
        self.lstm3 = nn.LSTM(
            input_size=hidden2 * 2,
            hidden_size=hidden3,
            batch_first=True,
            bidirectional=True
        )

        # 上一層為雙向，最後全連接層的輸入維度也要乘以 2
        self.fc = nn.Linear(hidden3 * 2, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        x, _ = self.lstm3(x)

        # 取最後一個時間步
        x = x[:, -1, :]
        x = self.fc(x)
        return x


# ----------------------------
# Early Stopping
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ----------------------------
# 單一 epoch 訓練
# ----------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


# ----------------------------
# 單一 epoch 驗證
# ----------------------------
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)

    return running_loss / len(loader.dataset)


# ----------------------------
# 預測
# ----------------------------
def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            preds.append(outputs.cpu().numpy())

    return np.vstack(preds)


# ----------------------------
# 1. 讀取與篩選資料
# ----------------------------
print("=" * 50)
print("載入資料...")

try:
    df = pd.read_csv(DATA_FILE)
    target_keywords = [f'B{i:04d}' for i in range(0, 19)]
    pattern = '|'.join(target_keywords)
    df_filtered = df[df['Battery_ID'].astype(str).str.contains(pattern)].copy()

    if len(df_filtered) > 0:
        df = df_filtered
        print(f"篩選成功，數據量: {len(df)} 筆")
    else:
        print("篩選後無資料，使用全部數據。")

    df = df.sort_values(['Battery_ID', 'Cycle']).reset_index(drop=True)
    df = df.dropna(subset=FEATURES + [TARGET, 'Battery_ID', 'Cycle']).copy()

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

# ----------------------------
# 2. 建立時序序列
# ----------------------------
print("\n" + "=" * 50)
print(f"建立時序序列 (window size = {SEQ_LEN})...")

all_sequences_X, all_sequences_y, all_sequences_meta = [], [], []
battery_list = sorted(df['Battery_ID'].unique())

for batt_id in battery_list:
    batt_data = df[df['Battery_ID'] == batt_id].sort_values('Cycle').reset_index(drop=True)
    if len(batt_data) < SEQ_LEN:
        continue

    feats = batt_data[FEATURES].values
    targ = batt_data[TARGET].values

    for i in range(SEQ_LEN - 1, len(batt_data)):
        all_sequences_X.append(feats[i - SEQ_LEN + 1: i + 1, :])
        all_sequences_y.append(targ[i])
        all_sequences_meta.append({
            'Battery_ID': batt_data.iloc[i]['Battery_ID'],
            'Cycle': batt_data.iloc[i]['Cycle']
        })

all_sequences_X = np.array(all_sequences_X, dtype=np.float32)
all_sequences_y = np.array(all_sequences_y, dtype=np.float32).reshape(-1, 1)

if len(all_sequences_X) == 0:
    print("無法產生序列，程式終止。")
    sys.exit(1)

print(f"總序列數: {len(all_sequences_X)}")

# ----------------------------
# 3. 針對不同 Dropout 進行連續實驗
# ----------------------------
for current_dropout in DROPOUT_LIST:
    print(f"\n\n{'★' * 50}")
    print(f"開始實驗：Dropout = {current_dropout}")
    print(f"{'★' * 50}")

    EXP_DIR = os.path.join(OUTPUT_DIR, f"Dropout_{current_dropout}")
    FIG_DIR = os.path.join(EXP_DIR, "figures")
    MODEL_DIR = os.path.join(EXP_DIR, "models")
    LOG_DIR = os.path.join(EXP_DIR, "logs")

    os.makedirs(EXP_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ----------------------------
    # 3.1 切分資料
    # ----------------------------
    X_train_temp, X_test_raw, y_train_temp, y_test_raw, meta_train_temp, meta_test = train_test_split(
        all_sequences_X, all_sequences_y, all_sequences_meta,
        test_size=TEST_RATIO, shuffle=True, random_state=RANDOM_STATE
    )

    n_features = X_train_temp.shape[2]
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    X_train_2d = X_train_temp.reshape(-1, n_features)
    X_test_2d = X_test_raw.reshape(-1, n_features)

    scaler_x.fit(X_train_2d)
    scaler_y.fit(y_train_temp)

    X_train_scaled = scaler_x.transform(X_train_2d).reshape(X_train_temp.shape)
    X_test = scaler_x.transform(X_test_2d).reshape(X_test_raw.shape)
    y_train_scaled = scaler_y.transform(y_train_temp)
    y_test = scaler_y.transform(y_test_raw)

    val_size = int(len(X_train_scaled) * VAL_RATIO)
    X_val = X_train_scaled[-val_size:]
    y_val = y_train_scaled[-val_size:]
    X_train = X_train_scaled[:-val_size]
    y_train = y_train_scaled[:-val_size]

    # DataLoader
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    test_dataset = SequenceDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    run_results = []

    # ----------------------------
    # 3.2 執行 N_RUNS 次獨立訓練
    # ----------------------------
    print(f"準備進行 {N_RUNS} 次獨立訓練 (每次最多 {MAX_EPOCHS} Epochs)...")

    for run_i in range(N_RUNS):
        print(f"執行 Run {run_i + 1}/{N_RUNS} ... ", end="")

        model = LSTMRegressor(
            input_size=n_features,
            hidden1=LSTM_UNITS_1,
            hidden2=LSTM_UNITS_2,
            hidden3=LSTM_UNITS_3,
            dropout=current_dropout
        ).to(DEVICE)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopper = EarlyStopping(patience=EARLY_STOP_PATIENCE)

        history = {
            'loss': [],
            'val_loss': []
        }

        train_start_time = time.time()

        actual_epochs = 0
        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss = validate_one_epoch(model, val_loader, criterion, DEVICE)

            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            actual_epochs += 1
            early_stopper.step(val_loss, model)

            if early_stopper.early_stop:
                break

        # 恢復最佳權重
        if early_stopper.best_state is not None:
            model.load_state_dict(early_stopper.best_state)

        train_end_time = time.time()
        train_duration = train_end_time - train_start_time

        test_start_time = time.time()

        y_pred_scaled = predict(model, test_loader, DEVICE)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_true = scaler_y.inverse_transform(y_test)

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        r2 = r2_score(y_true, y_pred)
        
        # --- 新增：計算該次預測誤差的標準差 ---
        error_std = np.std(y_true - y_pred) 
        
        lr = LEARNING_RATE

        test_end_time = time.time()
        test_duration = test_end_time - test_start_time

        print(f"實際 Epochs: {actual_epochs} | R²={r2:.4f}, RMSE={rmse:.4f}")
        print(f"  └ 耗時: 訓練 {train_duration:.2f}s | 測試 {test_duration:.4f}s")

        # 儲存圖表
        fig_path = os.path.join(FIG_DIR, f"run_{run_i + 1:02d}_epochs_{actual_epochs}.png")
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title(f'Run {run_i + 1} Loss Curve (Epochs: {actual_epochs})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        sorted_idx = np.argsort(y_true.flatten())[::-1]
        plt.plot(y_true.flatten()[sorted_idx], label='Actual SOH')
        plt.plot(y_pred.flatten()[sorted_idx], label='Predicted SOH')
        plt.title(f'Run {run_i + 1} Prediction (R2={r2:.3f})')
        plt.xlabel('Samples')
        plt.ylabel('SOH')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # 儲存模型
        model_path = os.path.join(MODEL_DIR, f"model_run_{run_i + 1:02d}.pth")
        torch.save(model.state_dict(), model_path)

        run_results.append({
            'dropout': current_dropout,
            'run_id': run_i + 1,
            'actual_epochs': actual_epochs,
            'max_epochs_setting': MAX_EPOCHS,
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'error_std': float(error_std), # --- 新增：將誤差標準差寫入報表 ---
            'best_val_loss': float(min(history['val_loss'])),
            'train_time_sec': float(train_duration),
            'test_time_sec': float(test_duration),
            'learning_rate': lr,
            'TIME_STEP': SEQ_LEN,
            'BATCH_SIZE': BATCH_SIZE,
            'lstm_units_1': LSTM_UNITS_1,
            'lstm_units_2': LSTM_UNITS_2,
            'lstm_units_3': LSTM_UNITS_3,
            'test_ratio': TEST_RATIO,
            'val_ratio': VAL_RATIO,
            'model_path': model_path
        })

    # ----------------------------
    # 3.4 匯出該 Dropout 的 CSV 摘要
    # ----------------------------
    run_results_df = pd.DataFrame(run_results)
    
    # --- 新增：計算這 15 次實驗 (N_RUNS) 表現的整體標準差 ---
    run_results_df['r2_std_across_runs'] = run_results_df['r2'].std()
    run_results_df['rmse_std_across_runs'] = run_results_df['rmse'].std()
    
    csv_path = os.path.join(EXP_DIR, f'Dropout_{current_dropout}_Summary.csv')
    run_results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

print("\n" + "=" * 50)
print(f"所有實驗 ({len(DROPOUT_LIST)} 種 Dropout 設定) 皆已順利執行完畢！")
print(f"請至資料夾查看結果: {OUTPUT_DIR}")
print("=" * 50)