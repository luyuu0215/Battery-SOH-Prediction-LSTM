import scipy.io
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.interpolate import interp1d
import os

# ==========================================
# 參數設定
# ==========================================
CV_VOLTAGE_THRES = 4.2      # CC 結束電壓
CUTOFF_CURRENT = 0.02       # CV 結束電流
IQR_MULTIPLIER = 1.5        # 離群值係數
ICA_MIN_POINTS = 15         # ICA 最少點數需求
FIXED_CC_CURRENT = 1.5      # [新增] 強制設定恆流值為 1.5A

class BatteryFeatureExtractor:
    def get_outlier_mask(self, data):
        """計算 IQR 並回傳非離群值遮罩"""
        valid_data = data[~np.isnan(data)]
        if len(valid_data) < 4:
            return np.ones(len(data), dtype=bool)

        Q1 = np.percentile(valid_data, 25)
        Q3 = np.percentile(valid_data, 75)
        IQR = Q3 - Q1
        D_low = Q1 - IQR_MULTIPLIER * IQR
        D_high = Q3 + IQR_MULTIPLIER * IQR
        
        return (data >= D_low) & (data <= D_high)

    def smooth_curve(self, y, window=11, poly=2):
        """平滑濾波"""
        try:
            if len(y) < window:
                window = len(y) if len(y) % 2 != 0 else len(y) - 1
            if window < 3: return y
            return signal.savgol_filter(y, window, poly)
        except:
            return y

    def process_cycle(self, time, voltage, current):
        # 1. 基礎過濾
        valid_idx = ~np.isnan(voltage) & ~np.isnan(current) & ~np.isnan(time)
        t, v, i = time[valid_idx], voltage[valid_idx], current[valid_idx]
        if len(t) < 10: return None

        # 2. 鎖定充電階段
        charge_mask = i > CUTOFF_CURRENT
        t_chg, v_chg, i_chg = t[charge_mask], v[charge_mask], i[charge_mask]
        if len(t_chg) < 10: return None
        t_chg = t_chg - t_chg[0]

        # 3. 切分 CC / CV
        over_limit = v_chg >= CV_VOLTAGE_THRES
        split_idx = np.argmax(over_limit) if np.any(over_limit) else len(v_chg) - 1
        
        t_cc, v_cc, i_cc = t_chg[:split_idx+1], v_chg[:split_idx+1], i_chg[:split_idx+1]
        t_cv, v_cv, i_cv = t_chg[split_idx+1:], v_chg[split_idx+1:], i_chg[split_idx+1:]
        
        # 將 CC 段電流強制設為定值 1.5A
        i_cc[:] = FIXED_CC_CURRENT
        
        # 4. CC 段離群值清洗
        if len(t_cc) > 10:
            dt_cc = np.diff(t_cc, prepend=t_cc[0])
            if len(dt_cc) > 1: dt_cc[0] = dt_cc[1]
            
            mask_v = self.get_outlier_mask(v_cc)
            mask_dt = self.get_outlier_mask(dt_cc)
            final_mask = mask_v & mask_dt
            
            if np.sum(final_mask) < 5:
                t_cc_cl, v_cc_cl = t_cc, v_cc
            else:
                t_cc_cl, v_cc_cl = t_cc[final_mask], v_cc[final_mask]
            
            # 不管感測器量到多少，直接生成一個全為 1.5 的陣列
            # 這樣計算容量和 ICA 時，就會完全依照 "定值 1.5A" 進行
            i_cc_cl = np.full_like(v_cc_cl, FIXED_CC_CURRENT)
        else:
            t_cc_cl, v_cc_cl = t_cc, v_cc
            i_cc_cl = np.full_like(v_cc, FIXED_CC_CURRENT)

        # 5. 特徵提取
        cc_time = t_cc_cl[-1] - t_cc_cl[0] if len(t_cc_cl) > 0 else 0
        cv_time = t_cv[-1] - t_cv[0] if len(t_cv) > 0 else 0
        
        cap_cc = 0
        if len(t_cc_cl) > 1:
            dt_cl = np.diff(t_cc_cl, prepend=t_cc_cl[0])
            cap_cc = np.sum(i_cc_cl * dt_cl) / 3600.0
            
        cap_cv = 0
        if len(t_cv) > 1:
            dt_cv = np.diff(t_cv, prepend=t_cv[0])
            cap_cv = np.sum(i_cv * dt_cv) / 3600.0
            
        ca_area = cap_cc + cap_cv

        # ICA
        ic_peak_h, ic_peak_v, ic_valley_mag = np.nan, np.nan, np.nan
        try:
            v_min, v_max = 3.3, 4.18
            ica_mask = (v_cc_cl >= v_min) & (v_cc_cl <= v_max)
            
            if np.sum(ica_mask) >= ICA_MIN_POINTS:
                v_ica = v_cc_cl[ica_mask]
                i_ica = i_cc_cl[ica_mask]
                t_ica = t_cc_cl[ica_mask]
                dt_ica = np.diff(t_ica, prepend=t_ica[0])
                q_ica = np.cumsum(i_ica * dt_ica) 
                
                v_smooth = self.smooth_curve(v_ica, window=11)
                q_smooth = self.smooth_curve(q_ica, window=11)
                
                v_grid = np.linspace(v_min, v_max, 400)
                v_uniq, idx_uniq = np.unique(v_smooth, return_index=True)
                q_uniq = q_smooth[idx_uniq]
                
                if len(v_uniq) > 5:
                    f_interp = interp1d(v_uniq, q_uniq, kind='linear', fill_value="extrapolate")
                    q_grid = f_interp(v_grid)
                    dqdv = np.gradient(q_grid, v_grid)
                    dqdv_smooth = self.smooth_curve(dqdv, window=21)
                    
                    peaks, _ = signal.find_peaks(dqdv_smooth, prominence=0.01)

                    if len(peaks) > 0:
                        peak_idx = peaks[0]
                        ic_peak_h = dqdv_smooth[peak_idx]
                        ic_peak_v = v_grid[peak_idx]

                    else:
                        # 如果真的很平找不到峰，才退回用全域最大值
                        peak_idx = np.argmax(dqdv_smooth)
                        ic_peak_h = dqdv_smooth[peak_idx]
                        ic_peak_v = v_grid[peak_idx]
                    
                    valleys, _ = signal.find_peaks(-dqdv_smooth)
                    if len(valleys) > 0:
                        valley_idx = valleys[np.argmax(-dqdv_smooth[valleys])]
                        ic_valley_mag = dqdv_smooth[valley_idx]
                    else:
                        ic_valley_mag = np.min(dqdv_smooth)
        except:
            pass 

        return {
            'IC_Peak_Height': ic_peak_h,
            'IC_Peak_Voltage': ic_peak_v,
            'IC_Valley_1_Magnitude': ic_valley_mag,
            'CA_Area': ca_area,
            'CC_Charging_Time': cc_time,
            'CV_Charging_Time': cv_time
        }

def process_mat_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing file: {file_path}")
        
    print(f"[INFO] Loading {file_path}...")
    mat = scipy.io.loadmat(file_path)
    extractor = BatteryFeatureExtractor()
    results = []
    
    battery_keys = [k for k in mat.keys() if k.startswith('B00') and 'cycle' in k]
    
    for key in battery_keys:
        print(f"Processing {key}...")
        try:
            cycles = mat[key][0, 0]['cycle']
            for i, c in enumerate(cycles.flatten()):
                if 'Voltage_measured' not in c.dtype.names: continue
                
                v = c['Voltage_measured'].flatten()
                curr = c['Current_measured'].flatten()
                t = c['Time'].flatten()
                
                feats = extractor.process_cycle(t, v, curr)
                
                if feats:
                    cap = c['Capacity'].flatten()[0] if 'Capacity' in c.dtype.names else np.nan
                    row = {'Battery_ID': key, 'Cycle': i + 1, **feats, 'Capacity': cap}
                    results.append(row)
        except Exception as e:
            print(f"[WARN] Error in {key}: {e}")

    df = pd.DataFrame(results)
    
    cols = ['Battery_ID', 'Cycle', 'IC_Peak_Height', 'IC_Peak_Voltage', 
            'IC_Valley_1_Magnitude', 'CA_Area', 'CC_Charging_Time', 
            'CV_Charging_Time', 'Capacity']
    for c in cols:
        if c not in df.columns: df[c] = np.nan
    df = df[cols]
    
    # ==========================================
    # [關鍵修改] 使用 線性插補 (Linear Interpolation)
    # ==========================================
    print(f"\n[INFO] 原始資料空值數: {df.isnull().sum().sum()}")
    
    # 定義插補函數
    def fill_missing(group):
        # 1. 確保按 Cycle 排序 (插補才有效)
        group = group.sort_values('Cycle')
        
        # 2. 線性插補 (處理中間空值)
        # limit_direction='inside' 表示只有當前後都有值時，中間的空值才會被補。
        group = group.infer_objects(copy=False)
        group = group.interpolate(method='linear', limit_area='inside')
        
        return group

    # 對每個 Battery_ID 分組執行插補
    df = df.groupby('Battery_ID', group_keys=False).apply(fill_missing)
    
    # ==========================================
    # [驗證代碼] 在刪除前，檢查 B00xx 的原始狀態
    # ==========================================
# ==========================================
# [驗證代碼] 在刪除前，檢查各電池的原始狀態
# ==========================================
    target_bats = ['B0005_cycleA', 'B0006_cycleA', 'B0007_cycleA', 'B0018_cycleA']

    for target_bat in target_bats:
        if target_bat in df['Battery_ID'].values:
            print(f"\n[DEBUG] 檢查 {target_bat} 在去頭去尾前的數據狀況:")
            check_df = df[df['Battery_ID'] == target_bat].sort_values('Cycle').head(45)
            print(check_df[['Cycle', 'IC_Peak_Height', 'Capacity']].to_string())
            print("-" * 50)
    else:
            print(f"\n[DEBUG] {target_bat} 不存在於資料中，跳過。")
    
    # 最後防線 (若整顆電池都壞掉，用全體平均補，機率極低)
    before_drop_len = len(df)
    df = df.dropna()
    after_drop_len = len(df)
    
    print(f"[INFO] 已切除頭尾數據。")
    print(f"刪除前數量: {before_drop_len}, 刪除後數量: {after_drop_len}")
    print(f"共移除了 {before_drop_len - after_drop_len} 筆頭尾異常數據")

  
    # ==========================================
    # [新增] SOH 計算 (Post-Interpolation)
    # 邏輯：SOH = 當前容量 / 初始容量 (該電池 Cycle 1 的容量)
    # ==========================================
    def calculate_soh(group):
        # 確保已排序
        group = group.sort_values('Cycle')
        # 取第 1 個週期的容量作為初始容量 (因為已做過 bfill，iloc[0] 必有值)
        init_cap = group['Capacity'].iloc[0]
        
        # 防呆：避免分母為 0
        if init_cap == 0:
            group['SOH'] = np.nan
        else:
            group['SOH'] = group['Capacity'] / init_cap
        return group

    print("[INFO] Calculating SOH...")
    df = df.groupby('Battery_ID', group_keys=False).apply(calculate_soh)
    
    return df

if __name__ == "__main__":
    # 請依據實際情況修改路徑
    FILE_NAME = r'C:\Users\luyu\Documents\MATLAB\matlabB0005~B0018.mat'
    OUTPUT_FILE = 'battery_features_final_linear_soh_B0005~B0018.csv'
    
    try:
        df_result = process_mat_file(FILE_NAME)
        df_result.to_csv(OUTPUT_FILE, index=False)
        print(f"\n[SUCCESS] 已輸出檔案 (含 SOH): {OUTPUT_FILE}")
        print(df_result[['Battery_ID', 'Cycle', 'Capacity', 'SOH']].head())
    except Exception as e:
        print(f"[FATAL ERROR] {e}")