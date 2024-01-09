% Generate axial strain recording

function record = generate_strain(type, src_loc, sta_loc, theta, t, stf_func, vel)

    % Note: The size of src_loc is N_src * 2 
    
    % Two nearby locations for finite difference
    % dr = 0.03/4*3;
    dr = vel / 30 / 10;
    sta1_loc = sta_loc + dr/2 * [cosd(theta), sind(theta)];
    sta2_loc = sta_loc - dr/2 * [cosd(theta), sind(theta)];
    
    % Displacement record for the two stations
    disp_rec1 = generate_disp(type, src_loc, sta1_loc, theta, t, stf_func, vel);
    disp_rec2 = generate_disp(type, src_loc, sta2_loc, theta, t, stf_func, vel);

    % Output component
    record = (disp_rec1 - disp_rec2) ./ dr;
end