
% Figure Reproduction Script
% This file reproduces Figures 2, 3, 4 from the manuscript

close all;
clear;

m=2^8;
n=2^12;
s=20;

ttau=zeros((log2(m)+1),1);
for i=1:(log2(m)+1)
    ttau(i)=2^(i-1);%[1,2,3,4,5,6,7,8,9,10:10:m/2,m];
end
average_time=10;

epoch_SDCD=zeros(1,length(ttau));
epoch_ASDCD=zeros(1,length(ttau));

CPU_SDCD=zeros(1,length(ttau));
CPU_ASDCD=zeros(1,length(ttau));

SDCD_epoch=zeros(average_time,length(ttau));
ASDCD_epoch=zeros(average_time,length(ttau));

SDCD_cputime=zeros(average_time,length(ttau));
ASDCD_cputime=zeros(average_time,length(ttau));

fprintf('length of t =%d\n',length(ttau))

for ii=1:length(ttau)
    tau=ttau(ii);

    for jj=1:average_time

        %%% Gaussian
        A=randn(m,n);
        fprintf('type of the matrix:  Gaussian\n')

        %%% Bernoulli
        %A=randi([0,1],m,n);
        %A(A==0)=-1;
        %fprintf('type of the matrix: Bernoulli\n')

        %%% Hadamard
        %A=hadamard(n);
        %A=A(randperm(n,m),:);
        %fprintf('type of the matrix: Hadmard\n')

        %% generated the right-hand vector b
        y=randn(m,1);
        Aty=A'*y;
        [value, ~] = select_kth_largest_abs_maxk(Aty, s);
        lambda=abs(value);
        x=max( 0, Aty - lambda ) - max( 0, -Aty - lambda );
        b=A*x;

        %% parameter setup
        opts.xstar=x;
        opts.TOL=10^(-6);
        opts.Max_iter=500000;
        opts.strategy=1;

        %%
        [xSDCD,OutSDCD]=mySDCD(A,b,lambda,tau,opts);

        [xASDCD,OutASDCD]=myASDCD(A,b,lambda,tau,opts);

        epoch_SDCD(ii)=epoch_SDCD(ii)+OutSDCD.iter/(m/tau);
        epoch_ASDCD(ii)=epoch_ASDCD(ii)+OutASDCD.iter/(m/tau);

        CPU_SDCD(ii)=CPU_SDCD(ii)+OutSDCD.times(end);
        CPU_ASDCD(ii)=CPU_ASDCD(ii)+OutASDCD.times(end);

        SDCD_epoch(jj,ii)=OutSDCD.iter/(m/tau);
        ASDCD_epoch(jj,ii)=OutASDCD.iter/(m/tau);
        %%%
        SDCD_cputime(jj,ii)=OutSDCD.times(end);
        ASDCD_cputime(jj,ii)=OutASDCD.times(end);

    end
    fprintf('ii=%d\n',ii)
end

%%%%%%
xlable=(log2(ttau))';

y1=SDCD_cputime';
miny1=min(y1');
maxy1=max(y1');
y1q25=quantile(y1,0.25,2);
y1q75=quantile(y1,0.75,2);

%%
y2=ASDCD_cputime';
miny2=min(y2');
maxy2=max(y2');
y2q25=quantile(y2,0.25,2);
y2q75=quantile(y2,0.75,2);


%%
figure
h = fill([xlable  fliplr(xlable)], [miny1 fliplr(maxy1)],'blue','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [y1q25' fliplr(y1q75')],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([xlable  fliplr(xlable)], [miny2 fliplr(maxy2)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([xlable  fliplr(xlable)], [y2q25' fliplr(y2q75')],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)
p1=plot( xlable, median(y1'), 'blue', 'LineWidth', 1,...
    'LineStyle', '-','Marker', 'o', 'DisplayName', 'SDCD');
p2=plot( xlable, median(y2'), 'magenta', 'LineWidth', 1,...
    'LineStyle', '-','Marker', 's', 'DisplayName', 'ASDCD');
ylabel('CPU','FontSize',15)
xlabel('$\log_{2}(\tau)$','Interpreter', 'latex','FontSize',15)
legend([p1 p2],{'SDCD','ASDCD'},'Interpreter', 'latex','location', 'best','FontSize',14)
txt=title( ['$m=$ ',num2str(m),', $n=$ ',num2str(n),', $s=$ ',num2str(s)]);
set(txt, 'Interpreter', 'latex','FontSize',17);



%%%%%%%
xlable=(log2(ttau))';

y1=SDCD_epoch';
miny1=min(y1');
maxy1=max(y1');
y1q25=quantile(y1,0.25,2);
y1q75=quantile(y1,0.75,2);

%%
y2=ASDCD_epoch';
miny2=min(y2');
maxy2=max(y2');
y2q25=quantile(y2,0.25,2);
y2q75=quantile(y2,0.75,2);


%%
figure
h = fill([xlable  fliplr(xlable)], [miny1 fliplr(maxy1)],'blue','EdgeColor', 'none');
set(h,'facealpha', .05)
hold on
h = fill([xlable  fliplr(xlable)], [y1q25' fliplr(y1q75')],'blue','EdgeColor', 'none');
set(h,'facealpha', .1)
h = fill([xlable  fliplr(xlable)], [miny2 fliplr(maxy2)],'magenta','EdgeColor', 'none');
set(h,'facealpha', .05)
h = fill([xlable  fliplr(xlable)], [y2q25' fliplr(y2q75')],'magenta','EdgeColor', 'none');
set(h,'facealpha', .1)
p1=semilogy( xlable, median(y1'), 'blue', 'LineWidth', 1,...
    'LineStyle', '-','Marker', 'o', 'DisplayName', 'SDCD');
p2=semilogy( xlable, median(y2'), 'magenta', 'LineWidth', 1,...
    'LineStyle', '-','Marker', 's', 'DisplayName', 'ASDCD');
ylabel('Epochs','FontSize',15)
xlabel('$\log_{2}(\tau)$','Interpreter', 'latex','FontSize',15)
legend([p1 p2],{'SDCD','ASDCD'},'Interpreter', 'latex','location', 'best','FontSize',14)
txt=title( ['$m=$ ',num2str(m),', $n=$ ',num2str(n),', $s=$ ',num2str(s)]);
set(txt, 'Interpreter', 'latex','FontSize',17);
