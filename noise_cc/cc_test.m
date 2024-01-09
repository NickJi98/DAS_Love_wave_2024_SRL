% Theoretical test of ambient noise cross-correlation

addpath('./matlab_fun');

set(groot, 'DefaultAxesFontsize', 18);
set(groot, 'DefaultAxesLineWidth', 1);
set(groot, 'DefaultLineLineWidth', 1);

%% Basic Parameters

% Wave type (R or L)
wave_type = 'L';

% Station information
% Distance from the origin (km)
sta_r = 0.5;
% Azimuth pointing from station 1 to 2 (deg)
sta_az = 0;
% Recording directions (deg)
sta_or = [0, 0];

% Noise source distribution
% Number of sources
N_src = 5000;
% Range of radius (km)
radius_range = [2, 3];
% Range of azimuth (rad)
azimuth_range = [0, 2*pi];

% Velocity (km/s)
vel = 2;

% Source time function
% Frequency (Hz)
fm = 20;  
% Function form
const = 2*pi^2*fm^2;
% stf_func = @(t) const .* t .* (const.*t.^2 - 3) .* exp(-const/2 .* t.^2);
stf_func = @(t) (1 - const .* t.^2) .* exp(-const/2 .* t.^2);
% stf_func = @(t) t .* exp(-const/2 .* t.^2);
% stf_func = @(t) exp(-const/2 .* t.^2);

%% Generate noise sources

% Generate random noise sources
src_phi = azimuth_range(1) + (azimuth_range(2)-azimuth_range(1)) .* rand(N_src, 1);
src_r = radius_range(1) + (radius_range(2)-radius_range(1)) .* rand(N_src, 1);
[src_phi, idx] = sort(src_phi);
src_r = src_r(idx);

% Equally distributed in azimuth
% src_phi = linspace(0, 2*pi, N_src)';
% src_r = radius_range(1) + (radius_range(2)-radius_range(1)) .* rand(N_src, 1);

% Convert to Cartesian
[src_x, src_y] = pol2cart(src_phi, src_r);
sta1_x = sta_r*cosd(sta_az+180); sta2_x = sta_r*cosd(sta_az);
sta1_y = sta_r*sind(sta_az+180); sta2_y = sta_r*sind(sta_az);

% Plot noise sources
figure('Name', 'Config');
scatter(src_x, src_y, 5, 'k', 'Filled');
xlabel('X [km]');   ylabel('Y [km]');   xlim([-3, 3]);
axis equal; hold on;

% Plot stations
dl = 0.2;
plot(sta1_x+dl*cosd(sta_or(1)).*[-1, 1], ...
    sta1_y+dl*sind(sta_or(1)).*[-1, 1], 'r-', 'LineWidth', 2); hold on;
plot(sta2_x+dl*cosd(sta_or(2)).*[-1, 1], ...
    sta2_y+dl*sind(sta_or(2)).*[-1, 1], 'r-', 'LineWidth', 2); hold on;
text(sta1_x, sta1_y+0.5, '1', 'FontSize', 18, 'HorizontalAlignment', 'center'); hold on;
text(sta2_x, sta2_y+0.5, '2', 'FontSize', 18, 'HorizontalAlignment', 'center');

%% Generate recordings

clear disp_rec1 disp_rec2 strain_rec1 strain_rec2;

% Time series
dt = 1/fm / 10;
t = 0:dt:2;

% Displacement recordings
disp_rec1 = generate_disp(wave_type, [src_x, src_y], [sta1_x, sta1_y], sta_or(1), t, stf_func, vel);
disp_rec2 = generate_disp(wave_type, [src_x, src_y], [sta2_x, sta2_y], sta_or(2), t, stf_func, vel);

% Strain recordings
strain_rec1 = generate_strain(wave_type, [src_x, src_y], [sta1_x, sta1_y], sta_or(1), t, stf_func, vel);
strain_rec2 = generate_strain(wave_type, [src_x, src_y], [sta2_x, sta2_y], sta_or(2), t, stf_func, vel);

%% Cross correlations

clear disp_cc strain_cc;
for i = 1:N_src
    [disp_cc(i, :), t_lag] = xcorr(disp_rec1(i, :), disp_rec2(i, :), 1/dt, 'biased');
end
t_lag = t_lag .* dt;
ave_disp_cc = mean(disp_cc, 1);

for i = 1:N_src
    [strain_cc(i, :), ~] = xcorr(strain_rec1(i, :), strain_rec2(i, :), 1/dt, 'biased');
end
ave_strain_cc = mean(strain_cc, 1);

%% Plot displacement profiles

% Traces to be plotted
plot_step = round(N_src / 250);
mask = 1:plot_step:N_src;

% Displacement
scale = 2 * max(max(abs(disp_rec1(mask, :))));
plot_profile(src_phi(mask), t, disp_rec1(mask, :), disp_rec2(mask, :), scale, 'Geophone');

% Disp. CC
scale = 2 * max(max(abs(disp_cc(mask, :))));
plot_cc_profile(src_phi(mask), t_lag, disp_cc(mask, :), scale, 'Geophone');

%% Plot strain profiles

% Displacement
scale = 2 * max(max(abs(strain_rec1(mask, :))));
plot_profile(src_phi(mask), t, strain_rec1(mask, :), strain_rec2(mask, :), scale, 'Fiber');

% Disp. CC
scale = 2 * max(max(abs(strain_cc(mask, :))));
plot_cc_profile(src_phi(mask), t_lag, strain_cc(mask, :), scale, 'Fiber');

%% Plot Green function

figure('Name', 'Extracted GF');
plot(t_lag, ave_disp_cc./max(abs(ave_disp_cc)), 'k-', 'LineWidth', 1.2); hold on;
plot(t_lag, ave_strain_cc./max(abs(ave_strain_cc)), 'r-', 'LineWidth', 1);
xlabel('Time Lag (s)');
ylabel('Normalized Amplitude');

% xline(2*sta_r/vel/sqrt(2), 'b-', 'LineWidth', 2);
% xline(-2*sta_r/vel/sqrt(2), 'b-', 'LineWidth', 2);

% title('Geophone');
legend('Geophone', 'Fiber', 'Location', 'north');