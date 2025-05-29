function xhat_EnKF = datafusion_KF_fireRobust(KalmanInput, varargin)
% datafusion_enKF  — Ensemble‑Kalman‑filter fusion with fire‑robust guards
%
%   xhat_EnKF = datafusion_enKF(KalmanInput, ...)
%
%  Robust extensions ported from datafusion_KF_fireRobust:
%    • Dynamic innovation gate (kFire)
%    • DTC‑referenced ΔT high threshold (deltaT_day / deltaT_night)
%    • Cross‑sensor consensus gate (epsConsensus)
%    • Outlier‑robust weights (trimPerc, trimmed‑mean for GK2A/S‑7)
%    • **NEW (Apr‑22)**  — GK2A pixels are *only* used when the collocated
%      Himawari Fire Mask Attribute (HimFMA) == 0  ➜ i.e. “not flagged fire”.
%
%  NAME–VALUE options (defaults):
%    'variance_base' , 2.5
%    'Q_process'     , 1
%    'N_ens'         , 51
%    'noiseScale'    , 1
%    'kFire'         , 9.2603
%    'deltaT_day'    , 85.556
%    'deltaT_night'  , 31.476
%    'epsConsensus'  , 9.1024
%    'trimPerc'      , 0.08
%    'covInfl'       , 1.0            (covariance‑inflation factor)
%    'weightCaps'    , [0.25 0.25 0.50]   % [Him GK2A S‑7]
%    'plot_measurement', 1|0
%
%  Returns:
%    xhat_EnKF  (epochs × 1)  – fused background temperature time‑series
%
%  Nur Fajar • Apr 2025
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 0.  INPUT PARSING
% -------------------------------------------------------------------------
ip = inputParser;
addParameter(ip,'variance_base',64.064);
addParameter(ip,'Q_process',0.0031062);
addParameter(ip,'N_ens',51);
addParameter(ip,'noiseScale',1.4141);
addParameter(ip,'kFire',4.5538);
addParameter(ip,'deltaT_day',59.017);
addParameter(ip,'deltaT_night',62.356);
addParameter(ip,'epsConsensus',21.444);
addParameter(ip,'trimPerc',0.18185);
addParameter(ip,'covInfl',1.462);
addParameter(ip,'weightCaps',[0.25 0.25 0.50]);  % [Him GK2A S‑7]
addParameter(ip,'plot_measurement',0);
parse(ip,varargin{:});
p = ip.Results;

% shorthands
N_ens         = p.N_ens;
variance_base = p.variance_base;
Q_process     = p.Q_process;
noiseScale    = p.noiseScale;
covInfl       = p.covInfl;
weightCap_him = p.weightCaps(1);
weightCap_gk2a= p.weightCaps(2);
weightCap_s7  = p.weightCaps(3);

% -------------------------------------------------------------------------
% 1.  PRE‑COMPUTE DTC AND FIRST VALID EPOCH
% -------------------------------------------------------------------------
DtcMIR = fillMissingDTC([KalmanInput.DtcMIR],[KalmanInput.Time]);
epochs = numel(KalmanInput);

hasValid = arrayfun(@(x) ...
          (~isnan(x.HimMIR)) || ...
          (~isempty(x.Gk2aMIR) && any(~isnan(x.Gk2aMIR))) || ...
          (~isempty(x.SentMIR_S7) && any(~isnan(x.SentMIR_S7))), ...
          KalmanInput);
firstValid = find(hasValid,1);
if isempty(firstValid)
    error('datafusion_enKF: no valid measurements in KalmanInput');
end

% -------------------------------------------------------------------------
% 2.  INITIALISE ENSEMBLE & STORAGE
% -------------------------------------------------------------------------
x0    = pickInitialValue(KalmanInput(firstValid));
x_ens = x0 + 0.1*randn(N_ens,1);

xhat_EnKF       = NaN(epochs,1);
X_ens_all       = NaN(N_ens,epochs);
xhat_EnKF(firstValid)   = mean(x_ens);
X_ens_all(:,firstValid) = x_ens;

% diagnostics
cntInnovGate = 0;   cntDeltaT = 0;   cntConsensus = 0;

% helpers
isDay = @(t) (hour(t)>=6 && hour(t)<18);
dtThr = @(dayFlag) p.deltaT_day*dayFlag + p.deltaT_night*(~dayFlag);

% -------------------------------------------------------------------------
% 3.  MAIN LOOP
% -------------------------------------------------------------------------
for k = firstValid+1 : epochs
    % ---------------- (A) FORECAST --------------------------------------
    if DtcMIR(k-1)~=0
        T = DtcMIR(k)/DtcMIR(k-1);
    else
        T = 1;
    end
    x_ens_pred = T*x_ens + sqrt(Q_process)*randn(N_ens,1);

    % ---------------- (B) CONSENSUS GATE (sensor‑level) -----------------
    % (GK2A representative is set to NaN if HimFMA≠0 even when data exist)
    repHim = KalmanInput(k).HimMIR;
    % ---------- GK2A validity incl. HimFMA test ----------
    use_firemask = KalmanInput(k).HimFMA == 0;  % 1 => “no fire flagged”
    gk2a_hasdata = ~isempty(KalmanInput(k).Gk2aMIR) && ...
                   any(~isnan(KalmanInput(k).Gk2aMIR));
    if use_firemask && gk2a_hasdata
        repGK = mean(KalmanInput(k).Gk2aMIR,'omitnan');
        gk2a_valid_considerFireMask = true;
    else
        repGK = NaN;
        gk2a_valid_considerFireMask = false;
    end
    repS7  = mean(KalmanInput(k).SentMIR_S7,'omitnan');

    reps  = [repHim repGK repS7];
    medRep= median(reps,'omitnan');

    him_valid = ~isnan(repHim) && abs(repHim - medRep) <= p.epsConsensus;
    gk_valid  = gk2a_valid_considerFireMask && ...
                abs(repGK  - medRep) <= p.epsConsensus;
    s7_valid  = ~isnan(repS7)  && abs(repS7  - medRep) <= p.epsConsensus;

    cntConsensus = cntConsensus + ...
                   sum(~[him_valid gk_valid s7_valid] & ~isnan(reps));

    % ---------------- (C) BUILD MEASUREMENT ARRAYS ----------------------
    z_all = [];   R_all = [];

    flagDay = isDay(KalmanInput(k).Time);
    % innovation OK?
    innovOK = @(meas,share) ...
        abs(meas - mean(x_ens_pred)) <= ...
        p.kFire * sqrt(var(x_ens_pred,1) + variance_base/share);

    % ---------- H I M A W A R I ----------------------------------------
    if him_valid
        sensorShare_him = weightCap_him / ...
            (weightCap_him*him_valid + weightCap_gk2a*gk_valid + ...
             weightCap_s7*s7_valid);
        meas = repHim;
        if     meas > DtcMIR(k) + dtThr(flagDay)
            cntDeltaT = cntDeltaT + 1;
        elseif ~innovOK(meas,sensorShare_him)
            cntInnovGate = cntInnovGate + 1;
        else
            z_all(end+1) = meas;
            R_all(end+1) = noiseScale * variance_base / sensorShare_him;
        end
    end

    % ---------- G K 2 A -------------------------------------------------
    if gk_valid
        pixVals = KalmanInput(k).Gk2aMIR(~isnan(KalmanInput(k).Gk2aMIR));
        % trimmed mean removal
        if numel(pixVals) > 2
            pixVals = sort(pixVals);
            drop = floor(p.trimPerc*numel(pixVals));
            pixVals = pixVals(1+drop : end-drop);
        end
        shareSensor = weightCap_gk2a / ...
            (weightCap_him*him_valid + weightCap_gk2a + weightCap_s7*s7_valid);
        sharePix = shareSensor / numel(pixVals);
        for vv = pixVals(:)'
            if     vv > DtcMIR(k) + dtThr(flagDay)
                cntDeltaT = cntDeltaT + 1;
            elseif ~innovOK(vv,sharePix)
                cntInnovGate = cntInnovGate + 1;
            else
                z_all(end+1) = vv;
                R_all(end+1) = noiseScale * variance_base / sharePix;
            end
        end
    end

    % ---------- S 3  S‑7 -----------------------------------------------
    if s7_valid
        pixVals = KalmanInput(k).SentMIR_S7(~isnan(KalmanInput(k).SentMIR_S7));
        if numel(pixVals) > 2
            pixVals = sort(pixVals);
            drop = floor(p.trimPerc*numel(pixVals));
            pixVals = pixVals(1+drop : end-drop);
        end
        shareSensor = weightCap_s7 / ...
            (weightCap_him*him_valid + weightCap_gk2a*gk_valid + weightCap_s7);
        sharePix = shareSensor / numel(pixVals);
        for vv = pixVals(:)'
            if     vv > DtcMIR(k) + dtThr(flagDay)
                cntDeltaT = cntDeltaT + 1;
            elseif ~innovOK(vv,sharePix)
                cntInnovGate = cntInnovGate + 1;
            else
                z_all(end+1) = vv;
                R_all(end+1) = noiseScale * variance_base / sharePix;
            end
        end
    end

    % ---------------- (D) ENKF UPDATE OR CARRY‑OVER ---------------------
    if isempty(z_all)
        x_ens = x_ens_pred;
    else
        m = numel(z_all);
        % perturbed observations
        Y_pred = repmat(x_ens_pred',m,1);
        Z_pert = repmat(z_all(:),1,N_ens) + ...
                 diag(sqrt(R_all))*randn(m,N_ens);

        y_bar = mean(Y_pred,2);
        x_bar = mean(x_ens_pred);

        X_dev = x_ens_pred - x_bar;
        Y_dev = Y_pred      - y_bar;

        P_xy  = (1/(N_ens-1)) * (X_dev') * (Y_dev');
        P_yy  = (1/(N_ens-1)) * (Y_dev*Y_dev') + diag(R_all);

        K_gain = P_xy / P_yy;

        x_ens = x_ens_pred + (K_gain * (Z_pert - Y_pred))';

        % covariance inflation
        if covInfl ~= 1
            mu = mean(x_ens);
            x_ens = mu + covInfl*(x_ens - mu);
        end
    end

    xhat_EnKF(k)       = mean(x_ens);
    X_ens_all(:,k)     = x_ens;
end

% -------------------------------------------------------------------------
% 4.  DIAGNOSTICS
% -------------------------------------------------------------------------
fprintf('\n========= ROBUST-EnKF DIAGNOSTICS =========\n');
fprintf('Innovation gate rejections : %d\n',cntInnovGate);
fprintf('ΔT threshold rejections    : %d\n',cntDeltaT);
fprintf('Consensus gate rejections  : %d\n',cntConsensus);
fprintf('-------------------------------------------\n');

if 0
    % collect differences
    diff_all   = [];
    labels_all = {};
    
    for k = 1:epochs
        xk = X_ens_all(k);
        rec = KalmanInput(k);
    
        % Himawari-8
        if ~isnan(rec.HimMIR)
            d = xk - rec.HimMIR;
            diff_all   = [diff_all; d];
            labels_all = [labels_all; {'Himawari-8'}];
        end
    
        % GK2A pixels
        if ~isempty(rec.Gk2aMIR)
            v = rec.Gk2aMIR(~isnan(rec.Gk2aMIR));
            d = xk - v(:);
            diff_all   = [diff_all; d];
            labels_all = [labels_all; repmat({'GK2A'},numel(d),1)];
        end
    
        % Sentinel-3 S7
        if ~isempty(rec.SentMIR_S7)
            v = rec.SentMIR_S7(~isnan(rec.SentMIR_S7));
            d = xk - v(:);
            diff_all   = [diff_all; d];
            labels_all = [labels_all; repmat({'Sentinel-3 S7'},numel(d),1)];
        end
    end
    
    % overall stats
    std_all  = std(diff_all,  'omitnan');
    rmse_all = sqrt(mean(diff_all.^2, 'omitnan'));
    fprintf('Overall Std Diff: %.4f K\n', std_all);
    fprintf('Overall RMSE    : %.4f K\n', rmse_all);
    
    % per-sensor stats
    sensors = unique(labels_all);
    for i = 1:numel(sensors)
        mask   = strcmp(labels_all, sensors{i});
        d_s    = diff_all(mask);
        std_i  = std(d_s,  'omitnan');
        rmse_i = sqrt(mean(d_s.^2, 'omitnan'));
        fprintf('%-15s Std: %.4f K   RMSE: %.4f K\n', sensors{i}, std_i, rmse_i);
    end
end
fprintf('===========================================\n');

if p.plot_measurement
    figure; plot(xhat_EnKF,'LineWidth',1.1); grid on;
    title('Robust EnKF fused background temperature');
    xlabel('Epoch'); ylabel('Temperature (K)');
end
end  % -- END main function -----------------------------------------------


% -------------------------------------------------------------------------
%                       H E L P E R  F N S
% -------------------------------------------------------------------------
function x0 = pickInitialValue(rec)
    if ~isnan(rec.HimMIR)
        x0 = rec.HimMIR;
    elseif ~isempty(rec.Gk2aMIR) && any(~isnan(rec.Gk2aMIR))
        x0 = rec.Gk2aMIR(find(~isnan(rec.Gk2aMIR),1));
    elseif ~isempty(rec.SentMIR_S7) && any(~isnan(rec.SentMIR_S7))
        x0 = rec.SentMIR_S7(find(~isnan(rec.SentMIR_S7),1));
    else
        error('pickInitialValue: record has no usable measurement');
    end
end
