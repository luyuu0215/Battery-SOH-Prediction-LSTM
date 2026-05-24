function batt_paired = pair_by_discharge(batt)
% PAIR_BY_DISCHARGE 
% 邏輯：
% 1. 遍歷所有數據，鎖定 'discharge' (放電)。
% 2. 遇到放電時，往回 (上一筆) 搜尋最近的 'charge'。
% 3. 忽略中間夾雜的 impedance (阻抗)。
% 4. 如果找不到對應的 charge，該筆 discharge 會被丟棄 (Drop)。
% 5. 如果最後一筆數據不是 discharge，自然不會被處理到 (Drop)。

    batt_paired = struct();
    % 預先建立一個空的 struct array，包含 charge 和 discharge 欄位
    batt_paired.cycle = struct('charge', {}, 'discharge', {});
    
    pair_count = 0;
    
    % 檢查輸入是否合法
    if ~isfield(batt, 'cycle')
        warning('Input data has no "cycle" field.');
        return;
    end
    
    cycles = batt.cycle;
    N = numel(cycles);
    
    % --- 主迴圈 ---
    for i = 1:N
        % 只有當前這筆是 'discharge' 時，我們才開始動作
        if strcmpi(cycles(i).type, 'discharge')
            
            current_discharge = cycles(i);
            found_charge = false;
            target_charge = [];
            
            % --- 往回搜尋 (Backwards Search) ---
            % 從 i-1 開始往回找，直到找到 charge 或撞到另一個 discharge
            for j = (i-1) : -1 : 1
                thisType = cycles(j).type;
                
                if strcmpi(thisType, 'charge')
                    % 找到了！這就是對應的充電
                    target_charge = cycles(j);
                    found_charge = true;
                    break; % 跳出往回找的迴圈
                    
                elseif strcmpi(thisType, 'discharge')
                    % 糟糕，往回找還沒看到 charge 就撞到上一個 discharge 了
                    % 代表這個 discharge 是孤兒 (Orphan)，沒有對應充電
                    break; 
                end
                % 如果是 impedance，迴圈會繼續往回找 (Skip)
            end
            
            % --- 配對成功，寫入結果 ---
            if found_charge
                pair_count = pair_count + 1;
                batt_paired.cycle(pair_count, 1).charge = target_charge;
                batt_paired.cycle(pair_count, 1).discharge = current_discharge;
            else
                % 沒找到 charge，這筆 discharge 直接 drop (不動作)
            end
        end
    end
    
    % 如果完全沒找到任何配對
    if pair_count == 0
        fprintf('Warning: No valid Charge-Discharge pairs found.\n');
    end
end