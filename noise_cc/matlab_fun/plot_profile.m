% Plot seismic profile

function plot_profile(src_phi, t, rec1, rec2, scale, name)
    
    % Number of sources
    Ns = length(src_phi);

    figure('Name', 'Profile');
    subplot(1,2,1);
    for i = 1:Ns
        trace = src_phi(i) + rec1(i, :) ./ scale;
        plot(t, trace, 'k-', 'LineWidth', 0.5); hold on;
    end
    xlabel('Time [s]'); ylabel('Source Azimuth [rad]');
    title(sprintf('%s 1', name));
    ylim([-pi/5, 2*pi+pi/5]);
    
    subplot(1,2,2);
    for i = 1:Ns
        trace = src_phi(i) + rec2(i, :) ./ scale;
        plot(t, trace, 'k-', 'LineWidth', 0.5); hold on;
    end
    xlabel('Time [s]'); ylabel('Source Azimuth [rad]');
    title(sprintf('%s 2', name));
    ylim([-pi/5, 2*pi+pi/5]);
end