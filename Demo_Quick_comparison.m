% Quick Solver Test Script
% This file is used to quickly run and compare the performance of different solvers

close all;
clear;

% Parameter settings
m = 2000;        % Number of matrix rows
n = 20000;       % Number of matrix columns  
s = 100;         % Sparsity level
tau = 100;       % Block size

%%% Generate test matrix (choose one) %%%

% Gaussian random matrix
A = randn(m, n);
fprintf('Matrix type: Gaussian\n')

% Bernoulli matrix (commented out)
%A = randi([0,1], m, n);
%A(A==0) = -1;
%fprintf('Matrix type: Bernoulli\n')

% Hadamard matrix (commented out)  
%A = hadamard(n);
%A = A(randperm(n, m), :);
%fprintf('Matrix type: Hadamard\n')

fprintf('Matrix dimensions: m = %8d, n = %8d, sparsity s = %4d, block size = %4d\n', ...
    m, n, s, tau)

%% Generate right-hand side vector b
y = randn(m, 1);
Aty = A' * y;
[value, ~] = select_kth_largest_abs_maxk(Aty, s);  % Set the value of mu
mu = abs(value);
x = max(0, Aty - mu) - max(0, -Aty - mu);
b = A * x;

%% Parameter setup for solvers
opts.xstar = x;        % True solution (for error calculation)
opts.TOL = 1e-12;      % Tolerance for convergence
opts.TOL1 = eps^2;     % Secondary tolerance
opts.strategy = 1;     % Strategy parameter

%% Run different solvers

% Stochastic Dual Coordinate Descent (SDCD)
[xSDCD, OutSDCD] = mySDCD(A, b, mu, tau, opts);

% Accelerated Stochastic Dual Coordinate Descent (ASDCD)  
[xASDCD, OutASDCD] = myASDCD(A, b, mu, tau, opts);

% Linearized Bregman (LB)
tic
normA2 = norm(A)^2;    % Compute squared norm of A
normtime = toc;

alpha = 2 / normA2;
alpha_mu = alpha / mu;
[xLB, OutLB] = myLinBreg(A, b, alpha_mu, mu, opts);

% Accelerated Linearized Bregman (ALB)
[xALB, OutALB] = myAceLinBreg(A, b, alpha_mu, mu, opts);

% Alternating Direction Method of Multipliers (ADMM)
beta = 0.01;
gamma = 0.99;
nu = 1 / normA2;
[xADMM, OutADMM] = myADMM(A, b, nu, beta, gamma, opts);

%% Display results

% CPU time comparison
fprintf('SDCD  CPU time: %8.5f s\n', OutSDCD.times(end))
fprintf('ASDCD CPU time: %8.5f s\n', OutASDCD.times(end)) 
fprintf('LB    CPU time: %8.5f s\n', OutLB.times(end))
fprintf('ALB   CPU time: %8.5f s\n', OutALB.times(end))
fprintf('ADMM  CPU time: %8.5f s\n', OutADMM.times(end))

fprintf('-------------------------------------\n')

% Relative squared error comparison
fprintf('ASDCD RSE: %8.2e\n', OutASDCD.error(end))
fprintf('LB    RSE: %8.2e\n', OutLB.error(end))
fprintf('ALB   RSE: %8.2e\n', OutALB.error(end))
fprintf('ADMM  RSE: %8.2e\n', OutADMM.error(end))

fprintf('-------------------------------------\n')
fprintf('CPU time to compute norm of A: %8.2e s\n', normtime)