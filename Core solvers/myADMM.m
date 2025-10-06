function [x,Out]=myADMM(A,b,tau,beta,gamma,opts)

% ADMM for solving basis pursuit problem
%    min \|x\|_1   s.t.       Ax=b
%
% the  initial x=0, hence A'r_0=A'(Ax_0-b)=-A'b
%
%Input: the coefficent matrix A, the vector b and opts
%opts.TOL: the stopping rule
%
%.....
%
%Output: the sparse solution x and Out
%
% Coded by Jiaxin Xie, Beihang University, xiejx@buaa.edu.cn
%

tic
[m,n]=size(A);

%% setting some parameter
flag=exist('opts');

if (flag && isfield(opts,'Max_iter'))
    Max_iter=opts.Max_iter;
else
    Max_iter=200000;
end

if (flag && isfield(opts,'TOL'))
    TOL=opts.TOL;
else
    TOL=10^-6;
end



if (flag && isfield(opts,'initial'))
    initialx=opts.initialx;
    initialy= opts.initialx;
else
    initialx=zeros(n,1);
    initialy=zeros(m,1);
end

if (flag && isfield(opts,'sparsity'))
    sparsity=opts.sparsity;
else
    sparsity=0;
end

x=initialx;
y=initialy;
Axb=A*x-b;

%%
if (flag && isfield(opts,'xstar'))
    xstar=opts.xstar;
    if m>=n
        normxstar=(norm(xstar-x))^2;
        error1=1;
        strategy=1;
    else
        strategy=0;
    end
else
    strategy=0;
end

if (flag && isfield(opts,'strategy'))
    strategy=opts.strategy;
    normxstar=(norm(xstar-x))^2;
end

if ~strategy
    normb=(norm(b))^2+1;
    error1=(norm(Axb))^2;
end

RSE(1)=1;

stopc=0;
iter=0;
tau_beta=tau/beta;
times(1)=toc;

while ~stopc
    tic
    iter=iter+1;
    sx=x-tau*(A'*(Axb-y/beta));
    x=max( 0, sx - tau_beta ) - max( 0, -sx - tau_beta);
    Axb=A*x-b;
    y=y-gamma*beta*(Axb);


    if strategy
        error1=(norm(x-xstar))^2/normxstar;
        RSE(iter+1)=error1;
        if error1<TOL  || iter>=Max_iter
            stopc=1;
        end
    else
        error1=(norm(A*x-b))^2/normb;
        RSE(iter+1)=error1;
        if  error1<TOL || iter>=Max_iter
            stopc=1;
        end
    end


    times(iter+1)=times(iter)+toc;
end
%%% setting Output
Out.error=RSE;
Out.iter=iter;
Out.times=times;

end