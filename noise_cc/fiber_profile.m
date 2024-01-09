% Theoretical test of DAS ambient noise cross-correlation

addpath('./matlab_fun');

set(groot, 'DefaultAxesFontsize', 24);
set(groot, 'DefaultAxesLineWidth', 1);
set(groot, 'DefaultLineLineWidth', 1.5);

%% Basic Parameters

%%% Station information
% N_sta * 3 matrix, each column setting x, y coordinates and orientation
% Distance in km, orientation in deg

% Number of station pairs
N_sta = 37;

% Station 1 information
% sta1_info = [-0.5*ones(N_sta,1), linspace(-1,1,N_sta)', 0*ones(N_sta,1)];
sta1_info = [-0.3*ones(N_sta,1), zeros(N_sta,1), linspace(0,180,N_sta)'];

% Station 2 information
sta2_info = repmat([0.3, 0, 45], [N_sta 1]);

%%% Noise source distribution
% Number of sources (Rayleigh and Love types)
% Figure 1 uses [3000, 3000]
N_src = [1000, 1000];
% Range of radius (km)
radius_range = [1.4, 1.6];
% Range of azimuth (rad)
azimuth_range = [0, 2*pi];

% Velocity (km/s) for Rayleigh and Love waves
vel = [0.3, 0.2];

% Central frequency (Hz)
fm = 4;  
% Source time function
const = 2*pi^2*fm^2;
% stf_func = @(t) const .* t .* (const.*t.^2 - 3) .* exp(-const/2 .* t.^2);
stf_func = @(t) (1 - const .* t.^2) .* exp(-const/2 .* t.^2);

%% Generate noise sources

% Generate random noise sources
% For better convergence, the azimuths are uniformly distributed
src_R_phi = linspace(azimuth_range(1), azimuth_range(2), N_src(1))';
src_R_r = radius_range(1) + (radius_range(2)-radius_range(1)) .* rand(N_src(1), 1);
src_L_phi = linspace(azimuth_range(1), azimuth_range(2), N_src(2))';
src_L_r = radius_range(1) + (radius_range(2)-radius_range(1)) .* rand(N_src(2), 1);

% Convert to Cartesian
[src_R_x, src_R_y] = pol2cart(src_R_phi, src_R_r);
[src_L_x, src_L_y] = pol2cart(src_L_phi, src_L_r);

% Plot noise sources
figure('Name', 'Config', 'Position', [0,0,720,420]);
scatter(src_R_x, src_R_y, 5, 'k', 'Filled'); hold on;
scatter(src_L_x, src_L_y, 5, 'g', 'Filled');
xlabel('X [km]');   ylabel('Y [km]');   ylim([-2, 2]);
axis equal; hold on;

% Plot stations
dl = 0.3;
for i = 1:6:N_sta
    plot(sta1_info(i,1)+dl*cosd(sta1_info(i,3)).*[-1, 1], ...
        sta1_info(i,2)+dl*sind(sta1_info(i,3)).*[-1, 1], 'r-', 'LineWidth', 1.5); hold on;
    plot(sta2_info(i,1)+dl*cosd(sta2_info(i,3)).*[-1, 1], ...
        sta2_info(i,2)+dl*sind(sta2_info(i,3)).*[-1, 1], 'b-', 'LineWidth', 1.5); hold on;
end
xlim([-3.5, 3.5]);  ylim([-2, 2]);

%% Generate recordings

% Time series
dt = 1/fm / 20;
t = 0:dt:10;

% Lag time series
tlag_max = 4;
t_lag = -tlag_max:dt:tlag_max;

clear ave_disp_cc ave_strain_cc;

for i = 1:N_sta

    % Displacement recordings
    disp_R_rec1 = generate_disp('R', [src_R_x, src_R_y], sta1_info(i,1:2), sta1_info(i,3), t, stf_func, vel(1));
    disp_R_rec2 = generate_disp('R', [src_R_x, src_R_y], sta2_info(i,1:2), sta2_info(i,3), t, stf_func, vel(1));
    disp_L_rec1 = generate_disp('L', [src_L_x, src_L_y], sta1_info(i,1:2), sta1_info(i,3), t, stf_func, vel(2));
    disp_L_rec2 = generate_disp('L', [src_L_x, src_L_y], sta2_info(i,1:2), sta2_info(i,3), t, stf_func, vel(2));

    % Strain recordings
    strain_R_rec1 = generate_strain('R', [src_R_x, src_R_y], sta1_info(i,1:2), sta1_info(i,3), t, stf_func, vel(1));
    strain_R_rec2 = generate_strain('R', [src_R_x, src_R_y], sta2_info(i,1:2), sta2_info(i,3), t, stf_func, vel(1));
    strain_L_rec1 = generate_strain('L', [src_L_x, src_L_y], sta1_info(i,1:2), sta1_info(i,3), t, stf_func, vel(2));
    strain_L_rec2 = generate_strain('L', [src_L_x, src_L_y], sta2_info(i,1:2), sta2_info(i,3), t, stf_func, vel(2));
    
    % Cross correlations
    for j = 1:N_src(1)
        disp_R_cc(j, :) = xcorr(disp_R_rec1(j, :), disp_R_rec2(j, :), tlag_max/dt, 'biased');
        strain_R_cc(j, :) = xcorr(strain_R_rec1(j, :), strain_R_rec2(j, :), tlag_max/dt, 'biased');
    end
    
    for j = 1:N_src(2)
        disp_L_cc(j, :) = xcorr(disp_L_rec1(j, :), disp_L_rec2(j, :), tlag_max/dt, 'biased');
        strain_L_cc(j, :) = xcorr(strain_L_rec1(j, :), strain_L_rec2(j, :), tlag_max/dt, 'biased');
    end
    
    ave_disp_cc(i, :) = mean(disp_R_cc, 1) + mean(disp_L_cc, 1);
    ave_strain_cc(i, :) = mean(strain_R_cc, 1) + mean(strain_L_cc, 1);
    
end

clear disp_R_rec1 disp_R_rec2 disp_L_rec1 disp_L_rec2 disp_R_cc disp_L_cc;
clear strain_R_rec1 strain_R_rec2 strain_L_rec1 strain_L_rec2 strain_R_cc strain_L_cc;

%% Plot cross correlations

% Y-axis quantity
% yaxis = sta1_info(:, 2);  yaxis_name = 'Y Location (km)';
yaxis = sta1_info(:, 3);  yaxis_name = 'Fiber 1 Orientation [deg]';

% Travel times
R = vecnorm((sta1_info(:, 1:2) - sta2_info(:, 1:2))');
tR = R ./ vel(1);  tL = R ./ vel(2);

% Additional scaling
scale = 2;

figure('Name', 'Profile', 'Position', [0,0,750,700]);
subplot(1,2,1);
max_amp = max(max(abs(ave_disp_cc)));
for i = 1:N_sta
        trace = yaxis(i) + ave_disp_cc(i, :) .* (scale*range(yaxis)/N_sta/max_amp);
        plot(t_lag, trace, 'k-', 'LineWidth', 1); hold on;
        scatter([-1 1]*tR(i), yaxis(i).*ones(1,2), 10, 'r', 'Filled'); hold on;
        scatter([-1 1]*tL(i), yaxis(i).*ones(1,2), 10, 'b', 'Filled'); hold on;
end
xlabel('Time Lag [s]'); ylabel(yaxis_name);
ylim([min(yaxis)-0.1*range(yaxis), max(yaxis)+0.1*range(yaxis)]);
title('Geophone');

subplot(1,2,2);
max_amp = max(max(abs(ave_strain_cc)));
for i = 1:N_sta
        trace = yaxis(i) + ave_strain_cc(i, :) .* (scale*range(yaxis)/N_sta/max_amp);
        plot(t_lag, trace, 'k-', 'LineWidth', 1); hold on;
        scatter([-1 1]*tR(i), yaxis(i).*ones(1,2), 10, 'r', 'Filled'); hold on;
        scatter([-1 1]*tL(i), yaxis(i).*ones(1,2), 10, 'b', 'Filled'); hold on;
end
xlabel('Time Lag [s]'); ylabel(yaxis_name);
ylim([min(yaxis)-0.1*range(yaxis), max(yaxis)+0.1*range(yaxis)]);
xlim([-4, 4]);
title('Fiber');

%% Analyze amplitude

% Window width
window = 1/(2*fm);
shift = 1/(8*fm);

% Record amplitude
AR = zeros(N_sta, 1);  AL = zeros(N_sta, 1);
for i = 1:N_sta
    
    mask = (t_lag > tR(i)-shift-window) & (t_lag < tR(i)-shift+window);
    tr = ave_strain_cc(i,:);  hilb_tr = abs(hilbert(tr));
    hilb_tr_R = hilb_tr(mask);
    [amp, ~] = max(hilb_tr_R);  AR(i) = amp;
    
    mask = (t_lag > tL(i)-shift-window) & (t_lag < tL(i)-shift+window);
    tr_L = tr(mask);  hilb_tr_L = hilb_tr(mask);
    [amp, ind] = max(hilb_tr_L);  AL(i) = sign(tr_L(ind)) * amp;
    
end

% Plot amplitude factor
figure('Name', 'Amplitude Factor', 'Position', [0,0,576,384]);
max_ampR = max(abs(AR));  max_ampL = max(abs(AL));
scatter(sta1_info(:, 3), AR./max_ampR, 30, 'r', 'Filled'); hold on;
scatter(sta1_info(:, 3), AL./max_ampR, 30, 'b', 'Filled'); hold on;

% Plot theoretical factor
theta1 = linspace(0, 180, 100);  theta2 = sta2_info(1,3);
AR0 = cosd(theta1).^2 .* cosd(theta2)^2;
AL0 = sind(2.*theta1) .* sind(2*theta2) ./ 4 .* (vel(1)/vel(2))^0.5;
max_amp0R = max(abs(AR0));  max_amp0L = max(abs(AL0));
plot(theta1, AR0./max_amp0R, 'k-', theta1, AL0./max_amp0R, 'k--');

xlabel('Fiber 1 Orientation [deg]');
ylabel('Relative Amplitude');
title(sprintf('Fiber 2 Orientation: %d deg', sta2_info(1,3)), 'FontSize', 22);
grid on;  xlim([0, 180]);  ylim([-1.1, 1.1]);
xticks([0 45 90 135 180]);
legend('Rayleigh', 'Love', 'Location', 'Southwest');