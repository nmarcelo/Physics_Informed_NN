% Example of Dr. Sam Raymond | Stanford University | sjray@stanford.edu
% Coded by Marcelo Yungaicela PhD

% Physics_Informed NN for a simple pendulum

%% 
clc
clear
% 

%% Dataset
new_dataset = false;

if new_dataset
    clc
    close
    % initial conditions
    theta_0Value = 0.1*pi:0.01*pi:pi/2;  % varing the initial heights
    k = length(theta_0Value);
    theta_t0Value = 0;
    
    for i=1:k
        [theta_real, theta_t_real] = simplePendulum(theta_0Value(i), theta_t0Value);
        if i==1
            N = length(theta_real);
            X_training = zeros(k*N,2);
        end
        X_training((i-1)*N+1:i*N,:) = [theta_real; theta_t_real]';
    end
    
    Y_training = X_training(2:end,:);
    X_training = X_training(1:end-1,:);

    save("Y_training.mat", "Y_training")
    save("X_training.mat", "X_training")
else
    load("Y_training.mat")
    load("X_training.mat")
end

%% Plotting
close all
plot(X_training(1:10,1),"--s","Marker","+"); hold on
plot(Y_training(1:10,1),"--b","Marker","o")

%% Training with convetional NN
training = false;

% layers
layers = [
    featureInputLayer(2,"Name","featureinput")
    fullyConnectedLayer(100,"Name","fc_1")
    tanhLayer("Name","tanh_1")
    batchNormalizationLayer("Name","batchnorm")
    fullyConnectedLayer(150,"Name","fc_3")
    tanhLayer("Name","tanh_3")
    dropoutLayer(0.25,"Name","dropout")
    fullyConnectedLayer(100,"Name","fc_4")
    tanhLayer("Name","tanh_2")
    fullyConnectedLayer(2,"Name","fc_2")
    regressionLayer("Name","regressionoutput")];

options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=0);

if training
    net = trainNetwork(X_training,Y_training,layers,options);
    save("net.mat", "net")
else
    load("net.mat")
end

%% training with P-I NN
clc
training_pinn = false;
% layers
layers = [
    featureInputLayer(2,"Name","featureinput")
    fullyConnectedLayer(100,"Name","fc_1")
    tanhLayer("Name","tanh_1")
    batchNormalizationLayer("Name","batchnorm")
    fullyConnectedLayer(150,"Name","fc_3")
    tanhLayer("Name","tanh_3")
    dropoutLayer(0.25,"Name","dropout")
    fullyConnectedLayer(100,"Name","fc_4")
    tanhLayer("Name","tanh_2")
    fullyConnectedLayer(2,"Name","fc_2")
    energyConsvLoss("dE")];

options = trainingOptions("adam", ...
    MaxEpochs=200, ...
    SequencePaddingDirection="left", ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=0);


XTrain = transpose(X_training);
YTrain = transpose(Y_training);

if training_pinn
    net_pinn = trainNetwork(X_training,Y_training,layers,options);
    save("net_pinn.mat", "net_pinn")
else
    load("net_pinn.mat")
end


%% Testing
% initial conditions
close all
theta_0Value = [pi/2 pi/3 pi/6];  % varing the initial heights
theta_t0Value = 0;  % Initially at rest

k = length(theta_0Value);

for i=1:k
    [theta_real, theta_t_real] = simplePendulum(theta_0Value(i), theta_t0Value);
    if i==1
        N = length(theta_real);
        X_testing = zeros(k*N,2);
    end
    X_testing((i-1)*N+1:i*N,:) = [theta_real; theta_t_real]';
end

Y_testing = X_testing(2:end,:);
X_testing = X_testing(1:end-1,:);

% conventional NN
Y_testing_predicted = predict(net,X_testing);

% PI NN
Y_testing_predicted_pinn = predict(net_pinn,X_testing);

figure,
subplot(1,2,1)
plot(Y_testing(:,1),Y_testing(:,2),"--s","Marker","x");  hold on
plot(Y_testing_predicted(:,1),Y_testing_predicted(:,2),"--b","Marker","x")
plot(Y_testing_predicted_pinn(:,1),Y_testing_predicted_pinn(:,2),"--r","Marker","x")

legend(["real", "conventional NN", "informed NN"])

% errors
error_conventional = [mse(Y_testing(:,1),Y_testing_predicted(:,1)) mse(Y_testing(:,2),Y_testing_predicted(:,2))];
error_pinn = [mse(Y_testing(:,1), Y_testing_predicted_pinn(:,1)) mse(Y_testing(:,2), Y_testing_predicted_pinn(:,2))];

subplot(1,2,2)
bar([error_conventional' error_pinn']); 
legend(["Conventional NN", "informed NN"])

%% definition of function for motion equation
function [theta_real, theta_t_real] = simplePendulum(theta_0Value, theta_t0Value)
    % Equations for the pendulum
    syms m a g theta(t)
    eqn = m*a == -m*g*sin(theta);
    
    syms r
    eqn = subs(eqn,a,r*diff(theta,2));
    
    eqn = isolate(eqn,diff(theta,2));
    
    syms omega_0
    eqn = subs(eqn,g/r,omega_0^2); % equation of motion for the pendulum
    
    syms x
    approx = taylor(sin(x),x,'Order',2);
    approx = subs(approx,x,theta(t));
    
    eqnLinear = subs(eqn,sin(theta(t)),approx);
    
    syms theta_0 theta_t0;
    
    theta_t = diff(theta);
    cond = [theta(0) == theta_0, theta_t(0) == theta_t0];
    assume(omega_0,'real')
    
    % solution of motion equation
    thetaSol(t) = dsolve(eqnLinear,cond);    
    
    gValue = 9.81;
    rValue = 3;
    omega_0Value = sqrt(gValue/rValue);
    T = 2*pi/omega_0Value;
    
    % initial conditions
    theta_0Value; % Solution only valid for small angles.
    theta_t0Value;      % Initially at rest
    
    vars   = [omega_0      theta_0      theta_t0];
    values = [omega_0Value theta_0Value theta_t0Value];
    thetaSolPlot = subs(thetaSol,vars,values);
    
    deltat=0.01*T;
    t = 0:deltat:10*T;
    
    theta_r=thetaSolPlot(t);
    
    % silly function
    theta_real= zeros(1,length(theta_r));
    theta_t_real = zeros(1,length(theta_r));
    theta_t_real(1)=theta_t0Value;
    for i=1:length(theta_real)
        theta_real(i) = round(theta_r(1,i),6);
    end
    
    theta_t_real(2:end)= diff(theta_real)/deltat;

end



