clc;

vars = who;

batteryNames = {};
for i = 1:numel(vars)
    v = vars{i};
    val = eval(v);

    if isstruct(val) && isfield(val,'cycle') && ~isempty(val.cycle) ...
            && isstruct(val.cycle) && isfield(val.cycle,'type')
        batteryNames{end+1} = v;
    end
end

fprintf("Found %d ORIGINAL battery variables in workspace\n", numel(batteryNames));

BatteryName     = strings(0);
OrigRecords     = [];
AfterPairing    = [];
AfterRebuild    = [];
FinalCycles     = [];

for i = 1:numel(batteryNames)
    bname = batteryNames{i};
    fprintf("\n========================================\n");
    fprintf("Processing %s ...\n", bname);

    batt = eval(bname);

    % --- 原始筆數 ---
    nOrig = numel(batt.cycle);
    fprintf("  [原始資料]        總 cycle 數 : %d\n", nOrig);

    % 原始資料中各類型數量
    types = {batt.cycle.type};
    nCharge    = sum(strcmpi(types, 'charge'));
    nDischarge = sum(strcmpi(types, 'discharge'));
    nImpedance = sum(strcmpi(types, 'impedance'));
    fprintf("    ├ charge     : %d\n", nCharge);
    fprintf("    ├ discharge  : %d\n", nDischarge);
    fprintf("    └ impedance  : %d\n", nImpedance);

    % --- Step 1: pairing ---
    batt_paired = pair_by_discharge(batt);
    nPaired = numel(batt_paired.cycle);
    nDropped_pairing = nDischarge - nPaired;
    fprintf("  [配對後 pair_by_discharge]\n");
    fprintf("    ├ 成功配對組數 : %d\n", nPaired);
    fprintf("    └ 配對失敗丟棄 : %d 筆 discharge\n", nDropped_pairing);

    % --- Step 2: rebuild ---
    batt_cycleA = rebuild_cycle_A(batt_paired, bname);
    nRebuilt = numel(batt_cycleA.cycle);
    nDropped_rebuild = nPaired - nRebuilt;
    fprintf("  [重建後 rebuild_cycle_A]\n");
    fprintf("    ├ 成功重建筆數 : %d\n", nRebuilt);
    fprintf("    └ 缺少 data 丟棄: %d 筆\n", nDropped_rebuild);

    % --- Step 4: 丟回 Workspace ---
    newName = bname + "_cycleA";
    assignin("base", newName, batt_cycleA);

    % --- 統計 ---
    BatteryName(end+1,1)  = string(bname);
    OrigRecords(end+1,1)  = nOrig;
    AfterPairing(end+1,1) = nPaired;
    AfterRebuild(end+1,1) = nRebuilt;
    FinalCycles(end+1,1)  = nRebuilt;
end

% --- 總覽表格 ---
fprintf("\n========================================\n");
fprintf("【總覽】各電池資料量變化\n");
fprintf("========================================\n");

summaryTbl = table(BatteryName, OrigRecords, AfterPairing, AfterRebuild, ...
    OrigRecords - AfterRebuild, ...
    'VariableNames', {'Battery', 'Original', 'AfterPairing', 'AfterRebuild', 'TotalDropped'});
disp(summaryTbl);

TOTAL_CYCLES = sum(FinalCycles);
fprintf("TOTAL_CYCLES (all batteries) = %d\n", TOTAL_CYCLES);
fprintf("TOTAL_DROPPED               = %d\n", sum(OrigRecords) - TOTAL_CYCLES);

clear batt bname i nOrig nPaired nRebuilt newName;
clear nCharge nDischarge nImpedance nDropped_pairing nDropped_rebuild types;