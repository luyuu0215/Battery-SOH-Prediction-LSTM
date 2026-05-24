# 鋰離子電池健康狀態 (SOH) 預測研究

本專案針對鋰離子電池進行健康狀態（State of Health, SOH）之預測。研究核心採用 LSTM 與 BiLSTM 雙向長短期記憶網路模型，並結合 增量容量分析 (ICA) 等特徵工程技術，以期將預測誤差（RMSE）控制在精準範圍內。

#開發流程與專案架構
本專案之開發分為兩個主要階段：資料前處理（MATLAB）與模型架構與訓練（VS Code / PyTorch）。
階段一：資料前處理與數據集監測 (MATLAB)
由於原始電池老化數據結構較為複雜，我們首先利用 MATLAB 進行資料清洗、特徵初步對齊與多個電池數據集的合併。
- 核心功能：負責將分散的電池循環數據（如 B0005、B0006、B0007、B0018 等）清洗並合併輸出為標準的 `.csv` 特徵矩陣。

階段二：特徵工程與時序模型預測 (VS Code / Python)
在 VS Code 開發環境下，使用 Python 與 PyTorch 框架進行核心演算法的實作。
- 特徵工程： 包含對增量容量（IC）曲線進行平滑化、提取 IC 峰值高度、峰值電壓、 Valley 值以及恆流/恆壓充電時間等關鍵時序特徵。
- 時序模型：建構並對比 LSTM 與 BiLSTM 的預測表現。
- 
檔案結構說明
```text
├── MATLAB_Data_Preprocessing/       # 階段一：MATLAB 資料處理
│   └── dataset_merge.m             # 數據集清洗與合併charge、discharge
│
├── VSCode_Deep_Learning/            # 階段二：VS Code 模型訓練
│   ├── data/
│   │   └── battery_features_final.csv  # 合併後的特徵數據集
│   ├── features_engineering.py     # 特徵工程、ICA 曲線提取與平滑化腳本
│   ├── train_lstm.py               # LSTM 模型訓練、多次獨立實驗與計時腳本
│   └── train_bilstm.py             # BiLSTM 模型雙向網路訓練腳本
│
└── README.md                       # 本說明文件
