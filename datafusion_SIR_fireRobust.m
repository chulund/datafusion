function xPF = datafusion_SIR_fireRobust(KalmanInput, varargin)
% datafusion_SIR_fireRobust  — sequential‑importance resampling (particle
%                              filter) with the *same* fire‑robust guards
%                              used in the KF, EnKF, and 4‑D‑Var versions.
%
%   xPF = datafusion_SIR_fireRobust(KalmanInput, ...)
%
% Robust ideas implemented
%   • GK2A pixels kept only when HimFMA == 0
%   • Cross‑sensor consensus gate             (epsConsensus)
%   • ΔT high‑threshold v. local DTC baseline (deltaT_day / deltaT_night)
%   • Dynamic innovation gate                 (kFire)
%   • Outlier‑robust trimmed‑mean for GK2A & S‑7 (trimPerc)
%
% NAME–VALUE OPTIONS             (defaults)
%   'Np'                , 51            – particle count
%   'processNoiseStd'   , 0.2           – σ of additive model noise
%   'variance_base'     , 1e‑3          – base measurement variance
%   'weightCaps'        , [0.25 0.25 0.50]   % [Him GK2A S‑7]
%   'plot_measurement'  , 1|0           – quick diagnostic line plot
%   ---------- robustness knobs ----------
%   'kFire'             , 9.2603
%   'deltaT_day'        , 85.556
%   'deltaT_night'      , 31.476
%   'epsConsensus'      , 9.1024
%   'trimPerc'          , 0.08
%
% OUTPUT
%   xPF  (epochs × 1)  – PF posterior mean (background temperature)
%
% Nur Fajar • Apr 2025
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% 0.  INPUT PARSING
% -------------------------------------------------------------------------
ip = inputParser;
addParameter(ip,'Np',51);
addParameter(ip,'processNoiseStd',0.41075);
addParameter(ip,'variance_base',40.674);
addParameter(ip,'weightCaps',[0.25 0.25 0.50]);   % [Him GK2A S‑7]
addParameter(ip,'plot_measurement',0);
% Robustness knobs
addParameter(ip,'kFire',5.5576);
addParameter(ip,'deltaT_day',53.345);
addParameter(ip,'deltaT_night',71.009);
addParameter(ip,'epsConsensus',15.165);
addParameter(ip,'trimPerc',0.13001);
parse(ip,varargin{:});
p = ip.Results;


Np              = p.Np;
processNoiseStd = p.processNoiseStd;
variance_base   = p.variance_base;
cap_him         = p.weightCaps(1);
cap_gk2a        = p.weightCaps(2);
cap_s7          = p.weightCaps(3);

% -------------------------------------------------------------------------
% 1.  PRE‑COMPUTE DTC AND RATIO MODEL
% -------------------------------------------------------------------------
[DtcMIR] = fillMissingDTC([KalmanInput.DtcMIR],[KalmanInput.Time]);
epochs   = numel(KalmanInput);

T_ratio  = ones(epochs,1);
for k = 2:epochs
    if DtcMIR(k-1)~=0
        T_ratio(k) = DtcMIR(k)/DtcMIR(k-1);
    end
end

% -------------------------------------------------------------------------
% 2.  BUILD ROBUST MEASUREMENTS (sensor caps + gates)
% -------------------------------------------------------------------------
[z_all, R_all, counts] = build_measurements_STS_fireRobust( ...
        KalmanInput, DtcMIR, epochs, ...
        cap_him, cap_gk2a, cap_s7, ...
        variance_base, ...
        p.kFire, p.deltaT_day, p.deltaT_night, ...
        p.epsConsensus, p.trimPerc);

% -------------------------------------------------------------------------
% 3.  INITIALISE PARTICLES
% -------------------------------------------------------------------------
x0          = pickInitialValue(KalmanInput(1));
spreadInit  = 1.0;
particles   = x0 + spreadInit*randn(Np,1);
weights     = (1/Np)*ones(Np,1);

xPF           = zeros(epochs,1);
allParticles  = NaN(epochs,Np);   %#ok<NASGU>

% -------------------------------------------------------------------------
% 4.  MAIN PF LOOP
% -------------------------------------------------------------------------
for k = 1:epochs
    % ---- (A) PREDICT ----------------------------------------------------
    ratio_k = T_ratio(k);
    particles = ratio_k*particles + processNoiseStd*randn(Np,1);

    % ---- (B) UPDATE -----------------------------------------------------
    z_k = z_all{k};   R_k = R_all{k};
    if ~isempty(z_k)
        for i=1:Np
            like = 1.0;
            x_i  = particles(i);
            for m=1:numel(z_k)
                diff = z_k(m) - x_i;
                like = like * exp(-0.5*diff^2 / R_k(m));
            end
            weights(i) = weights(i)*like;
        end
    end
    wSum = sum(weights);
    if wSum==0,  weights = (1/Np)*ones(Np,1);
    else,        weights = weights/wSum;
    end

    % ---- (C) RESAMPLE ---------------------------------------------------
    particles = resample_particles(particles,weights);
    weights   = (1/Np)*ones(Np,1);

    % ---- (D) SAVE POSTERIOR MEAN ---------------------------------------
    xPF(k) = mean(particles);
    allParticles(k,:) = particles'; %#ok<NASGU>
end

% -------------------------------------------------------------------------
% 5.  DIAGNOSTICS & PLOT
% -------------------------------------------------------------------------
fprintf('\n=========== ROBUST SIR diagnostics ===========\n');
fprintf('Consensus-gate rejections : %d\n',counts.consensus);
fprintf('ΔT threshold rejections   : %d\n',counts.deltaT);
fprintf('Innovation-gate rejects   : %d\n',counts.innov);
fprintf('----------------------------------------------\n');

if 0
    % collect differences
    diff_all   = [];
    labels_all = {};
    
    for k = 1:epochs
        xk = xPF(k);
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
        ds     = diff_all(mask);
        std_i  = std(ds,  'omitnan');
        rmse_i = sqrt(mean(ds.^2, 'omitnan'));
        fprintf('%-15s Std: %.4f K   RMSE: %.4f K\n', sensors{i}, std_i, rmse_i);
    end
end
fprintf('==============================================\n');

% optional plot
if p.plot_measurement
    figure; plot(xPF,'LineWidth',1.1); grid on;
    title('Robust SIR fused background temperature');
    xlabel('Epoch'); ylabel('Temperature (K)');
end


if p.plot_measurement
    figure; plot(xPF,'LineWidth',1.1); grid on;
    title('Robust SIR fused background temperature');
    xlabel('Epoch'); ylabel('Temperature (K)');
end
end % ================== END MAIN =========================================



% ========================================================================
%               R O B U S T   M E A S U R E M E N T   B U I L D E R
% ========================================================================
function [z_all,R_all,counts] = build_measurements_STS_fireRobust( ...
            KalmanInput, DtcMIR, epochs, ...
            cap_him, cap_gk2a, cap_s7, ...
            variance_base, ...
            kFire, dT_day, dT_night, epsCons, trimPerc)

z_all = cell(1,epochs);   R_all = cell(1,epochs);

counts.consensus = 0;
counts.deltaT    = 0;
counts.innov     = 0;

isDay = @(t) (hour(t)>=6 && hour(t)<18);
dtThr = @(flagDay) dT_day*flagDay + dT_night*(~flagDay);

for k = 1:epochs
    flagDay = isDay(KalmanInput(k).Time);

    % ------------- sensor representatives (apply HimFMA on GK2A) --------
    repHim = KalmanInput(k).HimMIR;

    useFmask = KalmanInput(k).HimFMA == 0;
    gk_has   = ~isempty(KalmanInput(k).Gk2aMIR) && ...
               any(~isnan(KalmanInput(k).Gk2aMIR));
    if useFmask && gk_has
        repGK = mean(KalmanInput(k).Gk2aMIR,'omitnan');
        gk_valid_pre = true;
    else
        repGK = NaN; gk_valid_pre = false;
    end
    repS7 = mean(KalmanInput(k).SentMIR_S7,'omitnan');

    reps   = [repHim repGK repS7];
    medRep = median(reps,'omitnan');

    him_valid = ~isnan(repHim) && abs(repHim-medRep)<=epsCons;
    gk_valid  =  gk_valid_pre   && abs(repGK -medRep)<=epsCons;
    s7_valid  = ~isnan(repS7)   && abs(repS7 -medRep)<=epsCons;

    counts.consensus = counts.consensus + ...
                       sum(~[him_valid gk_valid s7_valid] & ~isnan(reps));

    totalCap = cap_him*him_valid + cap_gk2a*gk_valid + cap_s7*s7_valid;
    if totalCap==0
        z_all{k} = []; R_all{k} = [];
        continue;
    end
    share_him  = cap_him  *him_valid / totalCap;
    share_gk2a = cap_gk2a *gk_valid  / totalCap;
    share_s7   = cap_s7   *s7_valid  / totalCap;

    z_k = []; R_k = [];

    % ---- HIM -----------------------------------------------------------
    if him_valid
        meas = repHim;
        if meas > DtcMIR(k)+dtThr(flagDay)
            counts.deltaT = counts.deltaT + 1;
        elseif abs(meas-DtcMIR(k)) > kFire*sqrt(variance_base/share_him)
            counts.innov = counts.innov + 1;
        else
            z_k(end+1) = meas;
            R_k(end+1) = variance_base/share_him;
        end
    end

    % ---- GK2A ----------------------------------------------------------
    if gk_valid
        vec = KalmanInput(k).Gk2aMIR(~isnan(KalmanInput(k).Gk2aMIR));
        if numel(vec)>2
            vec = sort(vec);
            drop = floor(trimPerc*numel(vec));
            vec = vec(1+drop : end-drop);
        end
        sharePix = share_gk2a / numel(vec);
        for vv = vec(:)'
            if vv > DtcMIR(k)+dtThr(flagDay)
                counts.deltaT = counts.deltaT + 1;
            elseif abs(vv-DtcMIR(k)) > kFire*sqrt(variance_base/sharePix)
                counts.innov = counts.innov + 1;
            else
                z_k(end+1) = vv;
                R_k(end+1) = variance_base/sharePix;
            end
        end
    end

    % ---- S‑7 -----------------------------------------------------------
    if s7_valid
        vec = KalmanInput(k).SentMIR_S7(~isnan(KalmanInput(k).SentMIR_S7));
        if numel(vec)>2
            vec = sort(vec);
            drop = floor(trimPerc*numel(vec));
            vec = vec(1+drop : end-drop);
        end
        sharePix = share_s7 / numel(vec);
        for vv = vec(:)'
            if vv > DtcMIR(k)+dtThr(flagDay)
                counts.deltaT = counts.deltaT + 1;
            elseif abs(vv-DtcMIR(k)) > kFire*sqrt(variance_base/sharePix)
                counts.innov = counts.innov + 1;
            else
                z_k(end+1) = vv;
                R_k(end+1) = variance_base/sharePix;
            end
        end
    end

    z_all{k} = z_k;
    R_all{k} = R_k;
end
end



% ========================================================================
%                        P A R T I C L E   U T I L S
% ========================================================================
function newParticles = resample_particles(oldParticles,weights)
Np = numel(oldParticles);
cdf = cumsum(weights);
newParticles = zeros(size(oldParticles));
step = 1/Np;
r = rand*step;
idx = 1;
for j = 1:Np
    thresh = r + (j-1)*step;
    while thresh > cdf(idx)
        idx = idx+1;
        if idx>Np, idx = Np; break; end
    end
    newParticles(j) = oldParticles(idx);
end
end



% ========================================================================
%                           M I S C
% ========================================================================
function x0 = pickInitialValue(rec)
if ~isnan(rec.HimMIR)
    x0 = rec.HimMIR;
elseif ~isempty(rec.Gk2aMIR) && any(~isnan(rec.Gk2aMIR))
    x0 = rec.Gk2aMIR(find(~isnan(rec.Gk2aMIR),1));
elseif ~isempty(rec.SentMIR_S7) && any(~isnan(rec.SentMIR_S7))
    x0 = rec.SentMIR_S7(find(~isnan(rec.SentMIR_S7),1));
else
    x0 = 300;
end
end
