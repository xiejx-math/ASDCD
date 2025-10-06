
% Figure Reproduction Script
% This file reproduces Figures 5, 6, 7 from the manuscript
% Compares performance of different algorithms
close all;
clear;

m=2000;
n=20000;
s=100;% sparsity
Max_iter=50000; % max iteration
tau=50; %size of the block
run_times=10; % average times

%%% the vector is used to store the numerical results
SDCD_error=zeros(run_times,Max_iter+1);
ASDCD_error=zeros(run_times,Max_iter+1);
LB_error=zeros(run_times,Max_iter+1);
ALB_error=zeros(run_times,Max_iter+1);
ADMM_error=zeros(run_times,Max_iter+1);
%%% epoach error
SDCD_epoch_error=zeros(run_times,Max_iter+1);
ASDCD_epoch_error=zeros(run_times,Max_iter+1);

%%%
CPU_SDCD=zeros(run_times,Max_iter+1);
CPU_ASDCD=zeros(run_times,Max_iter+1);
CPU_LB=zeros(run_times,Max_iter+1);
CPU_ALB=zeros(run_times,Max_iter+1);
CPU_ADMM=zeros(run_times,Max_iter+1);

for ii=1:run_times

    %%% Gaussian
    %A=randn(m,n);
    %fprintf('type of the matrix: Gaussian\n')

    %%% Bernoulli
    A=randi([0,1],m,n);
    A(A==0)=-1;
    fprintf('type of the matrix: Bernoulli\n')

    %%% Hadamard
    %A=hadamard(n);
    %A=A(randperm(n,m),:);
    %fprintf('type of the matrix: Hadamard\n')

    %% generated the right-hand vector b
    y=randn(m,1);
    Aty=A'*y;
    [value, ~] = select_kth_largest_abs_maxk(Aty, s);% set the value of mu
    mu=abs(value);
    x=max( 0, Aty - mu ) - max( 0, -Aty - mu );
    b=A*x;

    %%% parameter setup
    opts.xstar=x;
    opts.TOL=10^(-32);
    opts.TOL1=eps^2;
    opts.strategy=1;
    opts.Max_iter=Max_iter; % Number of iterations
    %%%
    [xSDCD,OutSDCD]=mySDCD(A,b,mu,tau,opts);

    %%%%%
    [xASDCD,OutASDCD]=myASDCD(A,b,mu,tau,opts);

    %%%% LinBreg
    opts.Max_iter=ceil(Max_iter*tau/m);
    alpha=2/norm(A)^2;
    alpha_mu=alpha/mu;
    [xLB,OutLB]=myLinBreg(A,b,alpha_mu,mu,opts);

    %%%% accelerated Linear Bregman iteration
    [xALB,OutALB]=myAceLinBreg(A,b,alpha_mu,mu,opts);

    %%%%% ADMM
    beta=0.01;
    gamma=0.99;
    nu=1/norm(A)^2;
    [xADMM,OutADMM]=myADMM(A,b,nu,beta,gamma,opts);

    %%%%CPU
    CPU_SDCD(ii,1:length(OutSDCD.times))=OutSDCD.times;
    CPU_ASDCD(ii,1:length(OutASDCD.times))=OutASDCD.times;
    CPU_LB(ii,1:length(OutLB.times))=OutLB.times;
    CPU_ALB(ii,1:length(OutALB.times))=OutALB.times;
    CPU_ADMM(ii,1:length(OutADMM.times))=OutADMM.times;
    %%% error
    SDCD_error(ii,1:length(OutSDCD.error))=OutSDCD.error;
    ASDCD_error(ii,1:length(OutASDCD.error))=OutASDCD.error;
    LB_error(ii,1:length(OutLB.error))=OutLB.error;
    ALB_error(ii,1:length(OutALB.error))=OutALB.error;
    ADMM_error(ii,1:length(OutADMM.error))=OutADMM.error;

    %%%
    xlableSDCD=[1:m/tau:OutSDCD.iter+1,OutSDCD.iter+1];
    SDCD_epoch_error(ii,1:length(xlableSDCD))=OutSDCD.error(round(xlableSDCD));
    xlableASDCD=[1:m/tau:OutASDCD.iter+1,OutASDCD.iter+1];
    ASDCD_epoch_error(ii,1:length(xlableASDCD))=OutASDCD.error(round(xlableASDCD));

    fprintf('Done,iter=%d\n',ii);
end


%% plot

%%%
xlable=1:(Max_iter+1);
%xlable=xlable';

%%%
y1=SDCD_epoch_error';
miny1=min(y1');
maxy1=max(y1');
y1q25=quantile(y1,0.25,2);
y1q75=quantile(y1,0.75,2);

%%%
y2=ASDCD_epoch_error';
miny2=min(y2');
maxy2=max(y2');
y2q25=quantile(y2,0.25,2);
y2q75=quantile(y2,0.75,2);

%%%
y3=LB_error';
miny3=min(y3');
maxy3=max(y3');
y3q25=quantile(y3,0.25,2);
y3q75=quantile(y3,0.75,2);

%%%
y4=ALB_error';
miny4=min(y4');
maxy4=max(y4');
y4q25=quantile(y4,0.25,2);
y4q75=quantile(y4,0.75,2);


%%%
y5=ADMM_error';
miny5=min(y5');
maxy5=max(y5');
y5q25=quantile(y5,0.25,2);
y5q75=quantile(y5,0.75,2);

%%%
marker_stepSDCD = 100;
marker_idxSDCD = 1:marker_stepSDCD:length(xlable);
marker_stepASDCD = 50;
marker_idxASDCD = 1:marker_stepASDCD:length(xlable);
marker_stepLB = 100;
marker_idxLB = 1:marker_stepLB:length(xlable);
marker_stepALB = 50;
marker_idxALB = 1:marker_stepALB:length(xlable);
marker_stepADMM = 100;
marker_idxADMM  = 1:marker_stepADMM:length(xlable);
%%
figure
h = fill([xlable  fliplr(xlable)], [miny5 fliplr(maxy5)],'red','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [y5q25' fliplr(y5q75')],'red','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([xlable  fliplr(xlable)], [miny3 fliplr(maxy3)],'black','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([xlable  fliplr(xlable)], [y3q25' fliplr(y3q75')],'black','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([xlable  fliplr(xlable)], [miny4 fliplr(maxy4)],'green','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([xlable  fliplr(xlable)], [y4q25' fliplr(y4q75')],'green','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([xlable  fliplr(xlable)], [miny2 fliplr(maxy2)],'blue','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([xlable  fliplr(xlable)], [y2q25' fliplr(y2q75')],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([xlable  fliplr(xlable)], [miny1 fliplr(maxy1)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([xlable  fliplr(xlable)], [y1q25' fliplr(y1q75')],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)
%%%
p1 = semilogy(xlable, median(y5'), 'Color', 'red', 'LineWidth', 1.2, ...
    'LineStyle', '-', 'DisplayName', 'ADMM', ...
    'Marker', 'o', 'MarkerIndices', marker_idxADMM , 'MarkerSize', 5);

p2 = semilogy(xlable, median(y3'), 'Color', 'black', 'LineWidth', 1.2, ...
    'LineStyle', '--', 'DisplayName', 'LB', ...
    'Marker', 's', 'MarkerIndices', marker_idxLB, 'MarkerSize', 5);

p3 = semilogy(xlable, median(y4'), 'Color', 'green', 'LineWidth', 1.2, ...
    'LineStyle', ':', 'DisplayName', 'ALB', ...
    'Marker', '^', 'MarkerIndices', marker_idxALB, 'MarkerSize', 5);

p4 = semilogy(xlable, median(y1'), 'Color', 'blue', 'LineWidth', 1.2, ...
    'LineStyle', '-.', 'DisplayName', 'SDCD', ...
    'Marker', 'd', 'MarkerIndices', marker_idxSDCD, 'MarkerSize', 5);

p5 = semilogy(xlable, median(y2'), 'Color', 'magenta', 'LineWidth', 1.2, ...
    'LineStyle', '-', 'DisplayName', 'ASDCD', ...
    'Marker', 'x', 'MarkerIndices', marker_idxASDCD, 'MarkerSize', 5);
set(gca, 'YScale', 'log')
ylim([10^(-12), 1])
xlim([1, 1000])
ylabel('RSE','Interpreter', 'latex')
xlabel('Epochs')
legend([ p1 p2 p3 p4 p5],{'ADMM','LB','ALB','SDCD','ASDCD'},'Interpreter', 'latex','location', 'best')
txt=title( ['$m=$ ',num2str(m),', $n=$ ',num2str(n),', $s=$ ',num2str(s)]);
set(txt, 'Interpreter', 'latex');

% Now plot CPU-Time vs RSE with shaded areas and sparse visual markers

% Set the marker spacing interval in seconds
interval = 1.8;

% Compute MarkerIndices from median CPU cost
marker_idx_ADMM = get_marker_indices(median(CPU_ADMM), interval);
marker_idx_LB = get_marker_indices(median(CPU_LB), interval);
marker_idx_ALB = get_marker_indices(median(CPU_ALB), interval);
marker_idx_SDCD = get_marker_indices(median(CPU_SDCD), interval);
marker_idx_ASDCD = get_marker_indices(median(CPU_ASDCD), interval);

%%%%%%%%%%%
y11=SDCD_error';
miny11=min(y11');
maxy11=max(y11');
y11q25=quantile(y11,0.25,2);
y11q75=quantile(y11,0.75,2);
%%%
y21=ASDCD_error';
miny21=min(y21');
maxy21=max(y21');
y21q25=quantile(y21,0.25,2);
y21q75=quantile(y21,0.75,2);
%%
figure
h = fill([median(CPU_ADMM) fliplr(median(CPU_ADMM) )], [miny5 fliplr(maxy5)],'red','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([median(CPU_ADMM) fliplr(median(CPU_ADMM) )], [y5q25' fliplr(y5q75')],'red','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([median(CPU_LB) fliplr(median(CPU_LB) )], [miny3 fliplr(maxy3)],'black','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([median(CPU_LB) fliplr(median(CPU_LB) )], [y3q25' fliplr(y3q75')],'black','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([median(CPU_ALB) fliplr(median(CPU_ALB))], [miny4 fliplr(maxy4)],'green','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([median(CPU_ALB) fliplr(median(CPU_ALB) )], [y4q25' fliplr(y4q75')],'green','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([median(CPU_SDCD) fliplr(median(CPU_SDCD) )], [miny11 fliplr(maxy11)],'blue','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([median(CPU_SDCD) fliplr(median(CPU_SDCD) )], [y11q25' fliplr(y11q75')],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([median(CPU_ASDCD) fliplr(median(CPU_ASDCD) )], [miny21 fliplr(maxy21)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([median(CPU_ASDCD) fliplr(median(CPU_ASDCD) )], [y21q25' fliplr(y21q75')],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)
% ADMM plot
p1 = semilogy(median(CPU_ADMM), median(y5'), 'Color', 'red', ...
    'LineWidth', 1.2, 'LineStyle', '-', 'DisplayName', 'ADMM', ...
    'Marker', 'o', 'MarkerIndices', marker_idx_ADMM, 'MarkerSize', 5);
% LB plot
p2 = semilogy(median(CPU_LB), median(y3'), 'Color', 'black', ...
    'LineWidth', 1.2, 'LineStyle', '--', 'DisplayName', 'LB', ...
    'Marker', 's', 'MarkerIndices', marker_idx_LB, 'MarkerSize', 5);
% ALB plot
p3 = semilogy(median(CPU_ALB), median(y4'), 'Color', 'green', ...
    'LineWidth', 1.2, 'LineStyle', ':', 'DisplayName', 'ALB', ...
    'Marker', '^', 'MarkerIndices', marker_idx_ALB, 'MarkerSize', 5);
% SDCD plot
p4 = semilogy(median(CPU_SDCD), median(y11'), 'Color', 'blue', ...
    'LineWidth', 1.2, 'LineStyle', '-.', 'DisplayName', 'SDCD', ...
    'Marker', 'd', 'MarkerIndices', marker_idx_SDCD, 'MarkerSize', 5);
% ASDCD plot
p5 = semilogy(median(CPU_ASDCD), median(y21'), 'Color', 'magenta', ...
    'LineWidth', 1.2, 'LineStyle', '-', 'DisplayName', 'ASDCD', ...
    'Marker', 'x', 'MarkerIndices', marker_idx_ASDCD, 'MarkerSize', 5);
% Set axis scales and limits
set(gca, 'YScale', 'log');
ylim([1e-12, 1]);
xlim([0, 18]);
% Labels and legend
xlabel('CPU', 'Interpreter', 'latex');
ylabel('RSE', 'Interpreter', 'latex');
legend([p1 p2 p3 p4 p5], {'ADMM', 'LB', 'ALB', 'SDCD', 'ASDCD'}, ...
    'Interpreter', 'latex', 'Location', 'best');
% Title with problem size
txt=title( ['$m=$ ',num2str(m),', $n=$ ',num2str(n),', $s=$ ',num2str(s)]);
set(txt, 'Interpreter', 'latex');


