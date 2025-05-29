function x_opt_global = datafusion_4DVar_fireRobust(KalmanInput, Nw, varargin)
% datafusion_4DVar_fireRobust  — sequential weak‑constraint 4‑D‑Var
%                                  with the same “fire‑robust” guards used
%                                  in datafusion_KF_fireRobust.
%
%   x_opt_global = datafusion_4DVar_fireRobust(KalmanInput, [Nw], ...)
%
% INPUTS
%   KalmanInput : 1×epochs struct array (same as other fusion codes)
%   Nw          : assimilation‑window length in epochs (‑1 ⇒ one big window)
%
% OPTIONAL NAME–VALUE PAIRS
%   'plot_measurement' , 1|0       (default 1)   — quick diagnostic plot
%   'Q_base'           , 1e‑3      (model‑error variance)
%   'maxIters'         , 100       (gradient‑descent iterations)
%   'stepSize'         , 2e‑4      (gradient‑descent learning rate)
%   'variance_base'    , 1e‑3      (base measurement variance)
%   'weightCaps'       , [0.25 0.25 0.50]   % [Him GK2A S7]
%   ---------- ROBUSTNESS KNOBS (same defaults as KF) ----------
%   'kFire'            , 9.2603    (innovation σ multiplier)
%   'deltaT_day'       , 85.556    (K above DTC baseline)
%   'deltaT_night'     , 31.476
%   'epsConsensus'     , 9.1024    (sensor‑level median gate)
%   'trimPerc'         , 0.08      (trimmed‑mean fraction for multi‑pixel)
%
% OUTPUT
%   x_opt_global : fused background temperature (epochs × 1)
%
% NOTES
%   •  GK2A pixels are **only** kept when the collocated Himawari Fire‑Mask
%      Attribute (HimFMA == 0).  This is identical to the robust KF.
%   •  Dynamic‑innovation gate uses a very simple prediction (DTC value),
%      because the 4‑D‑Var background state is not yet known when we build
%      the measurement list.  It still provides a large‑innovation guard.
%
% Nur Fajar • Apr 2025
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 0.  INPUT PARSING & CONSTANTS
% -------------------------------------------------------------------------
ip = inputParser;
addParameter(ip,'plot_measurement',0);
addParameter(ip,'Q_base',0.00062339);
addParameter(ip,'maxIters',100);
addParameter(ip,'stepSize',0.00018);
addParameter(ip,'variance_base',91.011);
addParameter(ip,'weightCaps',[0.25 0.25 0.50]);   % [Him GK2A S7]
% Robustness knobs
addParameter(ip,'kFire',3.2148);
addParameter(ip,'deltaT_day',95.11);
addParameter(ip,'deltaT_night',42.353);
addParameter(ip,'epsConsensus',9.9367);
addParameter(ip,'trimPerc',0.16534);
parse(ip,varargin{:});
p = ip.Results;

cap_him  = p.weightCaps(1);
cap_gk2a = p.weightCaps(2);
cap_s7   = p.weightCaps(3);

if nargin < 2,  Nw = 2;  end
fprintf('Running sequential 4‑D‑Var (robust)  |  window = %d epochs\n',Nw);

% -------------------------------------------------------------------------
% 1.  PRE‑COMPUTE DTC & TRANSITION RATIO
% -------------------------------------------------------------------------
[DtcMIR] = fillMissingDTC([KalmanInput.DtcMIR],[KalmanInput.Time]);
epochs   = numel(KalmanInput);

T_ratio  = ones(epochs,1);
for k=2:epochs
    if DtcMIR(k-1)~=0
        T_ratio(k) = DtcMIR(k)/DtcMIR(k-1);
    end
end

% -------------------------------------------------------------------------
% 2.  BUILD MEASUREMENTS WITH ROBUST GATES
% -------------------------------------------------------------------------
[z_all,R_all,diagCounts] = build_measurements_STS_fireRobust( ...
        KalmanInput, DtcMIR, epochs, ...
        cap_him, cap_gk2a, cap_s7, ...
        p.variance_base, ...
        p.kFire, p.deltaT_day, p.deltaT_night, ...
        p.epsConsensus, p.trimPerc);

% -------------------------------------------------------------------------
% 3.  PREP GLOBAL SOLUTION STORAGE
% -------------------------------------------------------------------------
x_opt_global       = zeros(epochs+1,1);
x_opt_global(1)    = pickInitialValue(KalmanInput(1));

if Nw == -1
    winStarts = 1;  winEnds = epochs;
else
    winStarts = 1 : Nw : epochs;
    winEnds   = min(winStarts + (Nw-1), epochs);
end
prevEndEpoch = 0;

% -------------------------------------------------------------------------
% 4.  MAIN WINDOW LOOP
% -------------------------------------------------------------------------
for iWin = 1:numel(winStarts)
    kStart = winStarts(iWin);
    kEnd   = winEnds(iWin);
    Nwin   = kEnd - kStart + 1;

    % ---- initial guess for this window
    x_guess = zeros(Nwin+1,1);
    if iWin==1
        x_guess(1) = x_opt_global(1);
    else
        x_guess(1) = x_opt_global(prevEndEpoch+1);
    end
    for kk = 1:Nwin
        x_guess(kk+1) = T_ratio(kStart+kk-1) * x_guess(kk);
    end

    % ---- subset data
    z_sub = z_all(kStart:kEnd);
    R_sub = R_all(kStart:kEnd);
    T_sub = T_ratio(kStart:kEnd);

    % ---- weak‑constraint 4‑D‑Var (simple GD)
    x_opt = x_guess;
    for iter = 1:p.maxIters
        [Jval,gradJ] = costFunction_and_Gradient_Weak4DVar( ...
                           x_opt, z_sub, R_sub, T_sub, p.Q_base);
        x_opt = x_opt - p.stepSize*gradJ;
        % (optional)  fprintf every 10 iter
    end

    % ---- write back to global array
    x_opt_global(kStart:kEnd+1) = x_opt;
    prevEndEpoch = kEnd;
end
x_opt_global = x_opt_global(2:end);

% -------------------------------------------------------------------------
% 5.  DIAGNOSTICS & OPTIONAL PLOT
% -------------------------------------------------------------------------
fprintf('\n=========== ROBUST 4-D-Var diagnostics ===========\n');
fprintf('Consensus-gate rejections : %d\n',diagCounts.consensus);
fprintf('ΔT threshold rejections   : %d\n',diagCounts.deltaT);
fprintf('Innovation-gate rejects   : %d\n',diagCounts.innov);
fprintf('-----------------------------------------------\n');

if 0
    % collect differences
    diff_all   = [];
    labels_all = {};
    
    for k = 1:epochs
        xk = x_opt_global(k);
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
fprintf('===================================================\n');

if p.plot_measurement
    figure;
    plot(x_opt_global,'LineWidth',1.1); grid on;
    title('Robust 4‑D‑Var fused background temperature');
    xlabel('Epoch'); ylabel('Temperature (K)');
end
end % ================= END MAIN ===========================================



% ========================================================================
%                      R O B U S T   M E A S U R E M E N T   B U I L D E R
% ========================================================================
function [z_all,R_all,counts] = build_measurements_STS_fireRobust( ...
            KalmanInput, DtcMIR, epochs, ...
            cap_him, cap_gk2a, cap_s7, ...
            variance_base, ...
            kFire, dT_day, dT_night, epsCons, trimPerc)

z_all = cell(1,epochs);
R_all = cell(1,epochs);

counts.consensus = 0;
counts.deltaT    = 0;
counts.innov     = 0;

isDay = @(t) (hour(t)>=6 && hour(t)<18);
dtThr = @(flagDay) dT_day*flagDay + dT_night*(~flagDay);

for k = 1:epochs
    flagDay = isDay(KalmanInput(k).Time);

    % ---------- sensor‑level representatives (+ HimFMA screening) --------
    repHim = KalmanInput(k).HimMIR;

    useFmask = KalmanInput(k).HimFMA == 0;
    gk_has   = ~isempty(KalmanInput(k).Gk2aMIR) && ...
               any(~isnan(KalmanInput(k).Gk2aMIR));
    if useFmask && gk_has
        repGK = mean(KalmanInput(k).Gk2aMIR,'omitnan');
        gk_valid = true;
    else
        repGK = NaN; gk_valid = false;
    end
    repS7 = mean(KalmanInput(k).SentMIR_S7,'omitnan');

    reps   = [repHim repGK repS7];
    medRep = median(reps,'omitnan');

    him_valid = ~isnan(repHim) && abs(repHim - medRep) <= epsCons;
    gk_valid  =  gk_valid       && abs(repGK  - medRep) <= epsCons;
    s7_valid  = ~isnan(repS7)   && abs(repS7  - medRep) <= epsCons;

    counts.consensus = counts.consensus + ...
                       sum(~[him_valid gk_valid s7_valid] & ~isnan(reps));

    totalCap = cap_him*him_valid + cap_gk2a*gk_valid + cap_s7*s7_valid;
    if totalCap==0
        z_all{k} = [];  R_all{k} = [];
        continue;
    end
    share_him  = cap_him  *him_valid / totalCap;
    share_gk2a = cap_gk2a *gk_valid  / totalCap;
    share_s7   = cap_s7   *s7_valid  / totalCap;

    z_meas = [];  R_meas = [];

    % ---------- HIMAWARI ------------------------------------------------
    if him_valid
        meas = repHim;
        if meas > DtcMIR(k) + dtThr(flagDay)
            counts.deltaT = counts.deltaT + 1;
        elseif abs(meas-DtcMIR(k)) > kFire*sqrt(variance_base/share_him)
            counts.innov  = counts.innov + 1;
        else
            z_meas(end+1) = meas;
            R_meas(end+1) = variance_base/share_him;
        end
    end

    % ---------- GK2A (multi‑pixel) --------------------------------------
    if gk_valid
        vec = KalmanInput(k).Gk2aMIR(~isnan(KalmanInput(k).Gk2aMIR));
        if numel(vec)>2
            vec = sort(vec);
            drop = floor(trimPerc*numel(vec));
            vec  = vec(1+drop : end-drop);
        end
        sharePix = share_gk2a / numel(vec);
        for vv = vec(:)'
            if vv > DtcMIR(k)+dtThr(flagDay)
                counts.deltaT = counts.deltaT + 1;
            elseif abs(vv-DtcMIR(k)) > kFire*sqrt(variance_base/sharePix)
                counts.innov  = counts.innov + 1;
            else
                z_meas(end+1) = vv;
                R_meas(end+1) = variance_base/sharePix;
            end
        end
    end

    % ---------- SENTINEL‑3 S7 ------------------------------------------
    if s7_valid
        vec = KalmanInput(k).SentMIR_S7(~isnan(KalmanInput(k).SentMIR_S7));
        if numel(vec)>2
            vec = sort(vec);
            drop = floor(trimPerc*numel(vec));
            vec  = vec(1+drop : end-drop);
        end
        sharePix = share_s7 / numel(vec);
        for vv = vec(:)'
            if vv > DtcMIR(k)+dtThr(flagDay)
                counts.deltaT = counts.deltaT + 1;
            elseif abs(vv-DtcMIR(k)) > kFire*sqrt(variance_base/sharePix)
                counts.innov  = counts.innov + 1;
            else
                z_meas(end+1) = vv;
                R_meas(end+1) = variance_base/sharePix;
            end
        end
    end

    z_all{k} = z_meas;
    R_all{k} = R_meas;
end
end



% ========================================================================
%                      C O S T   F U N C T I O N   &   G R A D I E N T
% ========================================================================
function [Jval,gradJ] = costFunction_and_Gradient_Weak4DVar( ...
                                    xVec, z_all, R_all, T_ratio, Qvar)
n            = numel(xVec);  Jval = 0;  gradJ = zeros(n,1);
epochsWindow = n-1;

% --- (A) measurement misfit --------------------------------------------
for k = 1:epochsWindow
    xm = xVec(k+1);
    zK = z_all{k};   Rk = R_all{k};
    for ii = 1:numel(zK)
        diff  = xm - zK(ii);
        Jval  = Jval + 0.5*diff^2 / Rk(ii);
        gradJ(k+1) = gradJ(k+1) + diff / Rk(ii);
    end
end

% --- (B) weak‑constraint model term -------------------------------------
for k = 2:epochsWindow
    diff = xVec(k+1) - T_ratio(k)*xVec(k);
    Jval = Jval + 0.5*diff^2 / Qvar;
    gradJ(k+1) = gradJ(k+1) + diff / Qvar;
    gradJ(k)   = gradJ(k)   - T_ratio(k)*diff / Qvar;
end
end



% ========================================================================
%                              M I S C
% ========================================================================
function x0 = pickInitialValue(rec)
if ~isnan(rec.HimMIR)
    x0 = rec.HimMIR;
elseif ~isempty(rec.Gk2aMIR) && any(~isnan(rec.Gk2aMIR))
    x0 = rec.Gk2aMIR(find(~isnan(rec.Gk2aMIR),1));
elseif ~isempty(rec.SentMIR_S7) && any(~isnan(rec.SentMIR_S7))
    x0 = rec.SentMIR_S7(find(~isnan(rec.SentMIR_S7),1));
else
    x0 = 300;               % safe fallback
end
end
