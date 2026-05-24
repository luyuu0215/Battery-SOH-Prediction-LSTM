function batt_paired = pair_charge_discharge_cycles(batt)
    % PAIR_CHARGE_DISCHARGE_CYCLES 
    % 過濾 NASA 電池數據，只保留 charge 和 discharge 類型
    % 並移除 impedance 測量數據

    batt_paired = struct();
    idx = 0;

    % 檢查原始 cycle 是否存在
    if ~isfield(batt, 'cycle')
        error('Input structure does not have a "cycle" field.');
    end

    for i = 1:numel(batt.cycle)
        thisType = batt.cycle(i).type;

        % 只保留充電 (charge) 和放電 (discharge)
        if strcmpi(thisType, 'charge') || strcmpi(thisType, 'discharge')
            idx = idx + 1;
            batt_paired.cycle(idx, 1) = batt.cycle(i);
        end
    end

    % 如果沒有找到任何 cycle，建立空結構以防報錯
    if idx == 0
        batt_paired.cycle = struct([]); 
    end
end