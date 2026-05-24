function batt_out = rebuild_cycle_A(batt_in, bname)
    % REBUILD_CYCLE_A (Flatten Version with .data fix)
    % 修正版：讀取 c.data.Voltage_measured 而不是 c.Voltage_measured

    batt_out = struct();
    
    if isfield(batt_in, 'cycle') && ~isempty(batt_in.cycle)
        N = numel(batt_in.cycle);
        
        % 為了加速，預先定義好欄位結構 (Optional)
        % 這裡直接跑迴圈
        
        valid_idx = 0; % 用來計算成功提取的筆數
        
        for i = 1:N
            % 取出配對好的 Charge 和 Discharge 結構
            c = batt_in.cycle(i).charge;
            d = batt_in.cycle(i).discharge;

            % --- 關鍵檢查：確保 data 欄位存在 ---
            % 如果這筆資料是壞的(沒有 data)，我們就跳過，避免報錯
            if ~isfield(c, 'data') || ~isfield(d, 'data')
                fprintf('Skipping cycle %d in %s (Missing data field)\n', i, bname);
                continue; 
            end

            valid_idx = valid_idx + 1;

            % --- 1. 識別欄位 (ID) ---
            batt_out.cycle(valid_idx,1).battery_id = string(bname);
            batt_out.cycle(valid_idx,1).cycle_id   = i;

            % --- 2. 充電數據 (Charge) ---
            % 注意：這裡加上了 .data
            batt_out.cycle(valid_idx,1).Voltage_measured     = c.data.Voltage_measured;
            batt_out.cycle(valid_idx,1).Current_measured     = c.data.Current_measured;
            batt_out.cycle(valid_idx,1).Temperature_measured = c.data.Temperature_measured;
            batt_out.cycle(valid_idx,1).Current_charge       = c.data.Current_charge;
            batt_out.cycle(valid_idx,1).Voltage_charge       = c.data.Voltage_charge;
            batt_out.cycle(valid_idx,1).Time                 = c.data.Time;

            % --- 3. 放電數據 (Discharge) ---
            % 注意：Capacity 通常也在 .data 裡面
            if isfield(d.data, 'Capacity')
                batt_out.cycle(valid_idx,1).Capacity = d.data.Capacity;
            else
                batt_out.cycle(valid_idx,1).Capacity = NaN;
            end
        end
    else
        batt_out.cycle = struct([]);
    end
end