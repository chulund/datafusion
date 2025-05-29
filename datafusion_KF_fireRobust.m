function xhat = datafusion_KF_fireRobust_weighted(KalmanInput, varargin)
% DATAFUSION_KF_FIREROBUST_WEIGHTED  — Kalman‑filter fusion with
% fire‑pixel exclusion **and** per‑pixel spatial/spectral/temporal weighting.
%
%   xhat = datafusion_KF_fireRobust_weighted(KalmanInput, ...)
%
% Combines the fire‑robust gates of *datafusion_KF_fireRobust* with the
% spectral–time–distance weighting pipeline of *datafusion_KF*.
% All legacy parameters are preserved; new per‑pixel weights are derived
% with helper functions:
%   • datafusionfunction_spectral2weight(λ_c , FWHM)
%   • datafusionfunction_time2weight(Δt)        (seconds → weight)
%   • datafusionfunction_distance2weight(d_pix) (pixel units → weight)
% The script prints a diagnostic summary at the end.

% -------------------------------------------------------------------------
% 0. USER‑TUNABLE PARAMETERS (defaults can be overridden via varargin)
% -------------------------------------------------------------------------
p = inputParser;
% legacy / weighting parameters
addParameter(p,'variance_base', 5.842);     % K² – will be rescaled per‑pixel
addParameter(p,'Qww',           0.001036);  % process‑noise variance
addParameter(p,'H_sys',         0.75315);   % system‑noise mapping
addParameter(p,'weightCaps',    []);        % [him gk2a s7]
% NEW fire‑robust parameters
addParameter(p,'kFire',         9.2603);    % innovation σ multiplier
addParameter(p,'deltaT_day',    85.556);    % K above DTC baseline (day)
addParameter(p,'deltaT_night',  31.476);    % K above DTC baseline (night)
addParameter(p,'epsConsensus',  9.1024);    % K consensus window
addParameter(p,'trimPerc',      0.079952);  % 8 % trimmed mean
parse(p,varargin{:});
pars = p.Results;

% -------------------------------------------------------------------------
% 1. SENSOR CAPS (global maximum share per epoch)
% -------------------------------------------------------------------------
weightCap_him  = 0.15814;  % slightly below equal share – tune as needed
weightCap_gk2a = 0.50434;
weightCap_s7   = 0.33788;
if ~isempty(pars.weightCaps)
    if numel(pars.weightCaps)~=3
        error('weightCaps must be 1×3');
    end
    weightCap_him = pars.weightCaps(1);
    weightCap_gk2a= pars.weightCaps(2);
    weightCap_s7  = pars.weightCaps(3);
end

% -------------------------------------------------------------------------
% 2. PRE‑COMPUTE DTC, INITIALISE ARRAYS
% -------------------------------------------------------------------------
[DtcMIR] = fillMissingDTC([KalmanInput.DtcMIR],[KalmanInput.Time]);
epochs   = numel(KalmanInput);

xhat    = zeros(1,epochs);
Qxx_arr = zeros(1,epochs)+0.0001;
Qxx     = 0.0001;
Qmm     = pars.H_sys*pars.Qww*pars.H_sys';

firstValid = find(~isnan([KalmanInput.HimMIR]),1);
if isempty(firstValid), error('No valid Himawari measurement'); end
xhat(1)   = KalmanInput(firstValid).HimMIR;

% weighting diagnostics
weights_him     = nan(1,epochs);
weights_gk2a    = cell(1,epochs);
weights_s7      = cell(1,epochs);

% Counters for diagnostics
countInnovGate=0; countDeltaT=0; countConsensus=0;

% helper lambdas
isDay = @(t) (hour(t)>=6 && hour(t)<18);

% -------------------------------------------------------------------------
% 3. MAIN LOOP
% -------------------------------------------------------------------------
for k = 2:epochs
    %----- 3.1 PREDICTION STEP -------------------------------------------
    if DtcMIR(k-1)~=0, T = DtcMIR(k)/DtcMIR(k-1); else, T=1; end
    xhat_pred  = T * xhat(k-1);
    Qxx_pred   = T^2*Qxx + Qmm;

    %----- 3.2 BUILD SENSOR VALIDITY FLAGS --------------------------------
    him_valid = ~isnan(KalmanInput(k).HimMIR);
    himFMA_ok = KalmanInput(k).HimFMA == 0;

    gk_vec  = KalmanInput(k).Gk2aMIR;
    gk_valid= himFMA_ok && ~isempty(gk_vec) && any(~isnan(gk_vec));

    s7_vec  = KalmanInput(k).SentMIR_S7;
    s7_valid= ~isempty(s7_vec) && any(~isnan(s7_vec));

    % If nothing at all, carry state forward
    if ~(him_valid||gk_valid||s7_valid)
        xhat(k)=xhat_pred; Qxx=Qxx_pred; Qxx_arr(k)=Qxx; continue;
    end

    %----- 3.3 CONSENSUS GATE (use median of reps) ------------------------
    reps=[]; if him_valid, reps(end+1)=KalmanInput(k).HimMIR; end
    if gk_valid,  reps(end+1)=mean(gk_vec,'omitnan'); end
    if s7_valid,  reps(end+1)=mean(s7_vec,'omitnan'); end
    medRep = median(reps,'omitnan');

    ok.him = ~(him_valid && abs(KalmanInput(k).HimMIR-medRep)>pars.epsConsensus);
    ok.gk  = ~(gk_valid  && abs(mean(gk_vec,'omitnan')-medRep)>pars.epsConsensus);
    ok.s7  = ~(s7_valid  && abs(mean(s7_vec,'omitnan')-medRep)>pars.epsConsensus);

    if him_valid && ~ok.him, countConsensus=countConsensus+1; end
    if gk_valid  && ~ok.gk,  countConsensus=countConsensus+sum(~isnan(gk_vec)); end
    if s7_valid  && ~ok.s7,  countConsensus=countConsensus+sum(~isnan(s7_vec)); end

    %----- 3.4 COMPUTE SENSOR SHARES --------------------------------------
    totalCap = weightCap_him*ok.him + weightCap_gk2a*ok.gk + weightCap_s7*ok.s7;
    share.him = weightCap_him*ok.him / totalCap;
    share.gk  = weightCap_gk2a*ok.gk  / totalCap;
    share.s7  = weightCap_s7 *ok.s7  / totalCap;

    %----- 3.5 INITIALISE MEASUREMENT VECTORS -----------------------------
    z = []; Rvec = [];

    % ΔT threshold for this timestamp
    deltaT = pars.deltaT_day * isDay(KalmanInput(k).Time) + ...
             pars.deltaT_night*~isDay(KalmanInput(k).Time);

    % ------------------------------------------------------------------
    % 3.6 HIMAWARI‑8 (single pixel)
    % ------------------------------------------------------------------
    if ok.him && share.him>0
        meas = KalmanInput(k).HimMIR;
        if meas > DtcMIR(k)+deltaT
            countDeltaT=countDeltaT+1;
        else
            innov = meas - xhat_pred;
            Sgate = Qxx_pred + pars.variance_base/share.him;
            if abs(innov) > pars.kFire*sqrt(Sgate)
                countInnovGate=countInnovGate+1;
            else
                % raw weight components
                w_raw = datafusionfunction_spectral2weight(3.8853,0.22) * 1 * 1; % λ_c & FWHM from AHI band 7
                w_final = share.him; % only one measurement so scaling trivial
                z(end+1)     = meas;
                Rvec(end+1)  = pars.variance_base / w_final;
                weights_him(k)= w_final;
            end
        end
    end

    % ------------------------------------------------------------------
    % 3.7 GK2A  (potentially multiple pixels)
    % ------------------------------------------------------------------
    if ok.gk && share.gk>0
        vec = gk_vec(~isnan(gk_vec));
        % Trim high/low extremes
        if numel(vec)>2
            vec = sort(vec);
            drop = floor(pars.trimPerc*numel(vec));
            vec  = vec(1+drop:end-drop);
        end
        raw_w = zeros(size(vec)); keepMask = true(size(vec));
        for ii=1:numel(vec)
            % ΔT / innovation gates first ------------------------------
            if vec(ii) > DtcMIR(k)+deltaT
                countDeltaT=countDeltaT+1; keepMask(ii)=false; continue;
            end
            innov = vec(ii)-xhat_pred;
            Sgate = Qxx_pred + pars.variance_base; % temp, will rescale later
            if abs(innov) > pars.kFire*sqrt(Sgate)
                countInnovGate=countInnovGate+1; keepMask(ii)=false; continue;
            end
            % Weight components if kept --------------------------------
            specW = datafusionfunction_spectral2weight(3.83,0.22);
            timeW = 1; % assume synchronous within one minute
            dist_km = deg2km(distance( ...
                    KalmanInput(k).HimLat, KalmanInput(k).HimLong, ...
                    KalmanInput(k).GK2ALat(ii), KalmanInput(k).GK2ALong(ii)));
            dist_pix = dist_km/1.4142;
            distW = datafusionfunction_distance2weight(dist_pix);
            raw_w(ii) = specW*timeW*distW;
        end
        vec      = vec(keepMask);
        raw_w    = raw_w(keepMask);
        if ~isempty(vec)
            w_final = raw_w / sum(raw_w) * share.gk;
            for jj = 1:numel(vec)
                z(end+1)    = vec(jj);
                Rvec(end+1) = pars.variance_base / w_final(jj);
            end
            weights_gk2a{k} = w_final;
        end
    end

    % ------------------------------------------------------------------
    % 3.8 SENTINEL‑3 S7
    % ------------------------------------------------------------------
    if ok.s7 && share.s7>0
        vec = s7_vec(~isnan(s7_vec));
        if numel(vec)>2
            vec = sort(vec);
            drop = floor(pars.trimPerc*numel(vec));
            vec  = vec(1+drop:end-drop);
        end
        raw_w = zeros(size(vec)); keepMask=true(size(vec));
        for ii=1:numel(vec)
            if vec(ii) > DtcMIR(k)+deltaT
                countDeltaT=countDeltaT+1; keepMask(ii)=false; continue;
            end
            innov = vec(ii)-xhat_pred;
            Sgate = Qxx_pred + pars.variance_base;
            if abs(innov) > pars.kFire*sqrt(Sgate)
                countInnovGate=countInnovGate+1; keepMask(ii)=false; continue;
            end
            specW = datafusionfunction_spectral2weight(3.74,0.22);
            % time difference (SentTimeOri_S7 assumed in KalmanInput) in seconds
            dt   = seconds(KalmanInput(k).SentTimeOri_S7 - KalmanInput(k).Time);     
            timeW= datafusionfunction_time2weight(dt);
            dist_km = deg2km(distance( ...
                    KalmanInput(k).HimLat, KalmanInput(k).HimLong, ...
                    KalmanInput(k).SentLatS7(ii), KalmanInput(k).SentLongS7(ii)));
            distW = datafusionfunction_distance2weight(dist_km/1.4142);
            raw_w(ii) = specW*timeW*distW;
        end
        vec   = vec(keepMask);
        raw_w = raw_w(keepMask);
        if ~isempty(vec)
            w_final = raw_w / sum(raw_w) * share.s7;
            for jj=1:numel(vec)
                z(end+1)    = vec(jj);
                Rvec(end+1) = pars.variance_base / w_final(jj);
            end
            weights_s7{k} = w_final;
        end
    end

    %----- 3.9 KALMAN UPDATE OR CARRY‑OVER -------------------------------
    if isempty(z)
        xhat(k)=xhat_pred; Qxx=Qxx_pred;
    else
        H = ones(numel(z),1);
        R = diag(Rvec);
        S = H*Qxx_pred*H' + R;
        K = Qxx_pred*H'/S;
        z = z(:);
        xhat(k) = xhat_pred + K*(z - H*xhat_pred);
        Qxx     = (1 - K*H)*Qxx_pred;
    end
    Qxx_arr(k)=Qxx;
end

xhat = xhat.';   % return column

% -------------------------------------------------------------------------
% 4. DIAGNOSTICS
% -------------------------------------------------------------------------
fprintf('\n====== FIRE‑ROBUST ⨯ WEIGHTED KF DIAGNOSTICS ======\n');
fprintf('Innovation‑gate rejections : %d\n',countInnovGate);
fprintf('\x0394T threshold rejections    : %d\n',countDeltaT);
fprintf('Consensus‑gate rejections  : %d\n',countConsensus);

% Average sensor shares (optional)
avg_w_gk2a = cellfun(@(c)mean(c,'omitnan'),weights_gk2a);
avg_w_s7   = cellfun(@(c)mean(c,'omitnan'),weights_s7);

fprintf('Mean Him share  : %.3f\n',mean(weights_him,'omitnan'));
fprintf('Mean GK2A share : %.3f\n',mean(avg_w_gk2a,'omitnan'));
fprintf('Mean S7 share   : %.3f\n',mean(avg_w_s7,'omitnan'));
fprintf('==================================================\n');

end