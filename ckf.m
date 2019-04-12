function [outputs] = ckf(options,obs)
%Classic Kalman Filter
%
%   Detailed explanation goes here
%

%unpack observation stuct
time    = obs.time;
y       = obs.meas;
m       = obs.num_meas;

%unpack options struct
ode_opts    = odeset('RelTol',options.tol,'AbsTol',options.tol');
n           = options.num_state;
R           = options.Rmat;
qmag        = options.qmag;

%preallocate
xi  = zeros(n+n^2,length(time));
STM = zeros(n,n,length(time));
Pi  = zeros(n,n,length(time));
Pim = zeros(n,n,length(time));
dxi = zeros(n,length(time));
ri_pre = zeros(m,length(time));
ri_post = zeros(m,length(time));
sim_meas = zeros(m,length(time));
sigma = zeros(n,length(time));
Qi = zeros(n,n,length(time));
Hi = [1, 0];

%Assign Initial
xi(:,1)     = options.IC_wPhi;
Pi(:,:,1)   = options.Pmat;
Pim(:,:,1)  = Pi(:,:,1);
STM(:,:,1)  = eye(n);

%condition on first observation @t=0
sim_meas(:,1) = options.obs_fcn(xi(1:n,1));
ri_pre(:,1) = y(:,1) - sim_meas(:,1);
Ki          = Pi(:,:,1)*Hi'*((Hi*Pi(:,:,1)*Hi' + R)\eye(m));

dxi(:,1)        = dxi(:,1) + Ki*(ri_pre(:,1) - Hi*dxi(:,1));
Pi(:,:,1)       = (eye(n) - Ki*Hi)*Pi(:,:,1)*(eye(n) - Ki*Hi)' + Ki*R*Ki';
ri_post(:,1)    = ri_pre(:,1) - Hi*dxi(:,1);
sigma(:,1)      = 2*sqrt(diag(Pi(:,:,1)));

for i = 2:length(y)
    
   %Integration from t_i-1 to t_i
    xi(:,i)  = [xi(1:n,i-1); reshape(eye(n),n^2,1)];
    tspan    = [time(i-1) time(i)];
    [~,xout] = ode45(options.integ_fcn, tspan, xi(:,i), ode_opts);
    xi(:,i)  = xout(end,:)';
    
    %Time Update
    Qi(:,:,i) = makeQ(time(i)-time(i-1),qmag);
    STM(:,:,i)  = reshape(xi(n+1:end,i), n, n);
    dxi(:,i)    = STM(:,:,i)*dxi(:,i-1);
    Pi(:,:,i)   = STM(:,:,i)*Pi(:,:,i-1)*STM(:,:,i)' + Qi(:,:,i);
    Pim(:,:,i)  = Pi(:,:,i);
    
    %Compute Observations
    sim_meas(:,i)    = options.obs_fcn(xi(1:n,i));
    ri_pre(:,i) = y(:,i)-sim_meas(:,i);
    Ki          = Pi(:,:,i)*Hi'*((Hi*Pi(:,:,i)*Hi' + R)\eye(m));
        
    %Measurement Update
    innov       = ri_pre(:,i) - Hi*dxi(:,i);
    dxi(:,i)    = dxi(:,i) + Ki*(innov);
    Pi(:,:,i)   = (eye(n) - Ki*Hi)*Pi(:,:,i)*(eye(n) - Ki*Hi)' + Ki*R*Ki';
    sigma(:,i)  = (2*sqrt(diag(Pi(:,:,i))));
    
    %post-update residuals      
    ri_post(:,i) = ri_pre(:,i) - Hi*dxi(:,i);
    
end

xtrue = options.xtrue;
x_err = xtrue(1:n,:) - (xi(1:n,:) + dxi);
rmsState = RMS_component(x_err(1:n,:));
rmsMeas = RMS_component(ri_post);

outputs.x_ckf   = xi;
outputs.dx_ckf  = dxi;
outputs.P_ckf   = Pi;
outputs.sigma_ckf   = sigma;
outputs.meas        = sim_meas;
outputs.resid_pre_ckf   = ri_pre;
outputs.resid_post_ckf  = ri_post;
outputs.STM         = STM;
outputs.Pm_ckf      = Pim;
outputs.rmsState    = rmsState;
outputs.H           = Hi;
outputs.rmsMeas     = rmsMeas;
%outputs.xRIC    =zRIC;

end

