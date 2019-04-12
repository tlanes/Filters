function [outputs] = ekf(options,obs)
%Extended Kalman Filter
%
% *calls @RMS_component for RMS values and @makeQ for process noise matrix     
%
%   EKF --
%       Inputs: 
%           options [struct]
%               .tol (integration tolerance)
%               .num_state (number of states)
%               .Rmat (sensor noise matrix)
%               .qmag (process noise magnitude)
%               .IC_wPhi (initial conditions with flat identity matrix)
%               .Pmat (state error covariance matrix)
%               .obs_fcn (observation/measurement function handle)
%               .integ_fcn (dynamics function handle for integration)
%               .xtrue (truth data)
%
%           obs [struct]
%               .time (time span vector)
%               .meas (observation/measurement data)
%               .num_meas (number of measurements at each step)
%
%


%unpack observation stuct
time    = obs.time;
y       = obs.meas;
st_id   = obs.station_id;

%unpack options struct
ode_opts    = odeset('RelTol',options.tol,'AbsTol',options.tol');
stationECI = options.stationECI;
n           = options.num_state;
R           = options.Rmat;
qmag        = options.qmag;

%preallocate
xi  = zeros(n+n^2,length(time));
Pi  = zeros(n,n,length(time));
ri_pre = zeros(2,length(time));
ri_post = zeros(3,length(time));
sigma = zeros(n,length(time));
Qi = zeros(n,n,length(time));

%Assign Initial
xi(:,1)     = options.IC_wPhi;
Pi(:,:,1)   = options.Pmat;

%condition on first observation @t=0
ri_pre(:,1) = y(:,1) - options.obs_fcn(xi(1:6,1), stationECI(1:6,1));
Hblk        = options.H_fcn_sc(xi(1:6,1), stationECI(1:6,1));
Ki          = Pi(:,:,1)*Hblk'*((Hblk*Pi(:,:,1)*Hblk' + R)\eye(2));

xi(1:n,1) = xi(1:n,1) + Ki*(ri_pre(:,1));
Pi(:,:,1)      = (eye(n) - Ki*Hblk)*Pi(:,:,1)*(eye(n) - Ki*Hblk)' + Ki*R*Ki';
ri_post(:,1) = [y(:,1) - options.obs_fcn(xi(1:6,1), stationECI(1:6,1));st_id(1)];
sigma(:,1)   = 3*sqrt(diag(Pi(:,:,1)));
%}

for i = 2:length(y)
    
    %Integration from t_i-1 to t_i
    tspan    = [time(i-1) time(i)];
    xi(:,i)  = [xi(1:n,i-1); reshape(eye(n),n^2,1)];
    [~,xout] = ode45(options.integ_fcn, tspan, xi(:,i), ode_opts);
    X        = xout(end,:)';
    xi(:,i)  = X;

    %make Q
    Qi(:,:,i)   = makeQ(time(i)-time(i-1),qmag);
    
    %Time Update
    STM       = reshape(X(n+1:end),n,n);
    Pi(:,:,i) = STM*Pi(:,:,i-1)*STM' + Qi(:,:,i);

    %
    %Compute Obs
    sim_meas = options.obs_fcn(xi(1:6,i), stationECI(1:6,i));
    ri_pre(:,i)   = y(:,i) - sim_meas;
    Hblk       = options.H_fcn_sc(xi(1:6,i), stationECI(1:6,i));
    Ki          = Pi(:,:,i)*Hblk'*((Hblk*Pi(:,:,i)*Hblk' + R)\eye(2));
   
    %Measurement Update
    xi(1:n,i)  = xi(1:n,i) + Ki*ri_pre(:,i);
    Pi(:,:,i)  = (eye(n) - Ki*Hblk)*Pi(:,:,i)*(eye(n) - Ki*Hblk)' + Ki*R*Ki';
    sigma(:,i) = 3*sqrt(diag(Pi(:,:,i)));
    %
    
    %post-fit residuals
    sim_meas_post = options.obs_fcn(xi(1:6,i), stationECI(1:6,i));
    ri_post(:,i)  = [y(:,i) - sim_meas_post;st_id(i)];
end

xtrue = options.xtrue;
x_err = xtrue(1:n,:) - xi(1:n,:);
rms3D_pos = RMS_3D(x_err(1:3,:));
rms3D_vel = RMS_3D(x_err(4:6,:));
rmsState = RMS_component(x_err);
rms_rho     = RMS_component(ri_post(1,:));
rms_rhod    = RMS_component(ri_post(2,:));

outputs.x_ekf   = xi;
outputs.P_ekf   = Pi;
outputs.sigma_ekf   = sigma;
outputs.resid_pre_ekf   = ri_pre;
outputs.resid_post_ekf  = ri_post;
outputs.rms3dpos    = rms3D_pos;
outputs.rms3dvel    = rms3D_vel;
outputs.rms_rho     = rms_rho;
outputs.rms_rhod    = rms_rhod;
outputs.rmsState    = rmsState;

end

