% Generate particle displacement recording

function record = generate_disp(type, src_loc, sta_loc, theta, t, stf_func, vel)

    % Note: The size of src_loc is N_src * 2 
    % Epicentral distance
    R = vecnorm((src_loc - sta_loc)')';
    
    if strcmp(type, 'R')
        % Add up two horizontal components
        coeff = (cosd(theta).*(sta_loc(1)-src_loc(:,1)) + sind(theta).*(sta_loc(2)-src_loc(:,2))) ./ (R .* sqrt(R));
        record = sqrt(vel) .* coeff .* stf_func(t-R./vel);
        
    elseif strcmp(type, 'L')
        % Add up two horizontal components
        coeff = -(cosd(theta).*(sta_loc(2)-src_loc(:,2)) - sind(theta).*(sta_loc(1)-src_loc(:,1))) ./ (R .* sqrt(R));
        record = sqrt(vel) .* coeff .* stf_func(t-R./vel);
        
    else
        error('Incorrect surface wave type: %s', type);
    end
end


function record = generate_disp_old(type, src_loc, sta_loc, theta, t, stf_func, vel)

    % Note: The size of src_loc is N_src * 2 
    % Epicentral distance
    R = vecnorm((src_loc - sta_loc)')';
    
    if strcmp(type, 'R')
        % Add up two horizontal components
        coeff = (cosd(theta).*(sta_loc(1)-src_loc(:,1)) + sind(theta).*(sta_loc(2)-src_loc(:,2))) ./ R.^2;
        record = coeff .* stf_func(t-R./vel);
        
    elseif strcmp(type, 'L')
        % Add up two horizontal components
        coeff = -(cosd(theta).*(sta_loc(2)-src_loc(:,2)) - sind(theta).*(sta_loc(1)-src_loc(:,1))) ./ R.^2;
        record = coeff .* stf_func(t-R./vel);
        
    else
        error('Incorrect surface wave type: %s', type);
    end
end