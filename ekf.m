function [outputs] = ekf(options,obs)
%Extended Kalman Filter
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

%preallocate
xi  = zeros(n+n^2,length(time));
Pi  = zeros(n,n,length(time));
ri_pre = zeros(m,length(time));
ri_post = zeros(m,length(time));
sim_meas = zeros(m,length(time));
sigma = zeros(n,length(time));
Qi = zeros(n,n,length(time));

%Assign Initial
xi(:,1)     = options.IC_wPhi;
Pi(:,:,1)   = options.Pmat;
qmag        = options.qmag;

%condition on first observation @t=0
sim_meas(:,1) = options.obs_fcn(xi(1:n,1));
ri_pre(:,1) = y(:,1) - sim_meas(:,1);
Hi          = options.H_fcn(xi(1:n,1));
Ki          = Pi(:,:,1)*Hi'*((Hi*Pi(:,:,1)*Hi' + R)\eye(m));

xi(1:n,1) = xi(1:n,1) + Ki*(ri_pre(:,1));
Pi(:,:,1)      = (eye(n) - Ki*Hi)*Pi(:,:,1)*(eye(n) - Ki*Hi)' + Ki*R*Ki';
ri_post(:,1) = y(:,1) - options.obs_fcn(xi(1:n,1));
sigma(:,1)   = 2*sqrt(diag(Pi(:,:,1)));
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

    %Compute Obs
    sim_meas(:,i)   = options.obs_fcn(xi(1:n,i));
    ri_pre(:,i)     = y(:,i) - sim_meas(:,i);
    Hi              = options.H_fcn(xi(1:n,1));    
    Ki              = Pi(:,:,i)*Hi'*((Hi*Pi(:,:,i)*Hi' + R)\eye(m));
   
    %Measurement Update
    xi(1:n,i)  = xi(1:n,i) + Ki*ri_pre(:,i);
    Pi(:,:,i)  = (eye(n) - Ki*Hi)*Pi(:,:,i)*(eye(n) - Ki*Hi)' + Ki*R*Ki';
    sigma(:,i) = 2*sqrt(diag(Pi(:,:,i)));
    
    %post-fit residuals
    sim_meas(:,i) = options.obs_fcn(xi(1:n,i));
    ri_post(:,i)  = y(:,i) - sim_meas(:,i);
end

xtrue = options.xtrue;
x_err = xtrue(1:n,:) - xi(1:n,:);
rmsState = RMS_component(x_err);
rmsMeas  = RMS_component(ri_post);

outputs.meas    = sim_meas;
outputs.x_ekf   = xi;
outputs.P_ekf   = Pi;
outputs.sigma_ekf   = sigma;
outputs.resid_pre_ekf   = ri_pre;
outputs.resid_post_ekf  = ri_post;
outputs.rmsState    = rmsState;
outputs.rmsMeas     = rmsMeas;


end

