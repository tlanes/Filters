function [outputs] = ckf(options,obs)
%Classic Kalman Filter
%
%   CKF --
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
stationECI  = options.stationECI;
n           = options.num_state;
R           = options.Rmat;

%preallocate
xi  = zeros(n+n^2,length(time));
STM = zeros(n,n,length(time));
Pi  = zeros(n,n,length(time));
Pim = zeros(n,n,length(time));
dxi = zeros(n,length(time));
ri_pre = zeros(2,length(time));
ri_post = zeros(3,length(time));
sigma = zeros(n,length(time));
Qi = zeros(n,n,length(time));
Hi = zeros(2,n,length(time));

%Assign Initial
xi(:,1)     = options.IC_wPhi;
Pi(:,:,1)   = options.Pmat;
Pim(:,:,1)  = Pi(:,:,1);
STM(:,:,1)  = eye(n);

%condition on first observation @t=0
Hblk        = Htilde_sc_rho_rhod_Lanes(xi(1:n,1),stationECI(:,1));
Hi(:,:,1)   = Hblk;
ri_pre(:,1) = y(:,1) - options.obs_fcn(xi(1:n,1), stationECI(1:n,1));
Ki          = Pi(:,:,1)*Hblk'*((Hblk*Pi(:,:,1)*Hblk' + R)\eye(2));

dxi(:,1)        = dxi(:,1) + Ki*(ri_pre(:,1) - Hblk*dxi(:,1));
Pi(:,:,1)       = (eye(n) - Ki*Hblk)*Pi(:,:,1)*(eye(n) - Ki*Hblk)' + Ki*R*Ki';
ri_post(:,1)    = [ri_pre(:,1) - Hblk*dxi(:,1);st_id(1)];
sigma(:,1)      = 3*sqrt(diag(Pi(:,:,1)));

for i = 2:length(y)
    
   %Integration from t_i-1 to t_i
    xi(:,i)  = [xi(1:n,i-1); reshape(eye(n),n^2,1)];
    tspan    = [time(i-1) time(i)];
    [~,xout] = ode45(options.integ_fcn, tspan, xi(:,i), ode_opts);
    xi(:,i)  = xout(end,:)';
    
    %Time Update
    STM(:,:,i)  = reshape(xi(n+1:end,i), n, n);
    dxi(:,i)    = STM(:,:,i)*dxi(:,i-1);
    Pi(:,:,i)   = STM(:,:,i)*Pi(:,:,i-1)*STM(:,:,i)';
    Pim(:,:,i)  = Pi(:,:,i);
    
    %Compute Observations
    sim_meas    = options.obs_fcn(xi(1:n,i),stationECI(1:n,i));
    ri_pre(:,i) = y(:,i)-sim_meas;
    Hblk        = options.H_fcn_sc(xi(1:n,i),stationECI(:,i));
    Hi(:,:,i)   = Hblk;
    Ki          = Pi(:,:,i)*Hblk'*((Hblk*Pi(:,:,i)*Hblk' + R)\eye(2));
        
    %Measurement Update
    innov       = ri_pre(:,i) - Hblk*dxi(:,i);
    dxi(:,i)    = dxi(:,i) + Ki*(innov);
    Pi(:,:,i)   = (eye(n) - Ki*Hblk)*Pi(:,:,i)*(eye(n) - Ki*Hblk)' + Ki*R*Ki';
    sigma(:,i)  = (3*sqrt(diag(Pi(:,:,i))));
    
    %post-update residuals      
    ri_post(:,i) = [ri_pre(:,i) - Hblk*dxi(:,i);st_id(i)];
    
end

rms_rho     = RMS_component(ri_post(1,:));
rms_rhod    = RMS_component(ri_post(2,:));

xtrue = options.xtrue;
x_err = xtrue(1:n,:) - (xi(1:n,:) + dxi);
rmsState = RMS_component(x_err(1:n,:));
rms3D_pos = RMS_3D(x_err(1:3,:));
rms3D_vel = RMS_3D(x_err(4:6,:));

outputs.x_ckf   = xi;
outputs.dx_ckf  = dxi;
outputs.P_ckf   = Pi;
outputs.sigma_ckf   = sigma;
outputs.resid_pre_ckf   = ri_pre;
outputs.resid_post_ckf  = ri_post;
outputs.rms3dpos    = rms3D_pos;
outputs.rms3dvel    = rms3D_vel;
outputs.rms_rho     = rms_rho;
outputs.rms_rhod    = rms_rhod;
outputs.STM         = STM;
outputs.Pm_ckf      = Pim;
outputs.rmsState    = rmsState;
outputs.H           = Hi;
%outputs.xRIC    =zRIC;

end

