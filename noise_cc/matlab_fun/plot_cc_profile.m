% Plot cross-correlation profile

function plot_cc_profile(src_phi, t, cc_rec, scale, name)
    
    % Number of sources
    Ns = length(src_phi);

    figure('Name', 'CC Profile');
    for i = 1:Ns
        trace = src_phi(i) + cc_rec(i, :) ./ scale;
        plot(t, trace, 'k-', 'LineWidth', 0.5); hold on;
    end
    xlabel('Time Lag [s]'); ylabel('Source Azimuth [rad]');
    title(sprintf('%s', name));
    ylim([-pi/5, 2*pi+pi/5]);
    
end