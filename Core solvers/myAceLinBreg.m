function [x,Out]=myAceLinBreg(A,b,alpha,mu,opts)

% Nesterov accelerated linearized Bregman iteration for solving 
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
% references:
% [1] Huang, B., Ma, S. & Goldfarb, D.
% Accelerated Linearized Bregman Method. J Sci Comput 54, 428–453 (2013). 
% https://doi.org/10.1007/s10915-012-9592-9

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
    initialv=opts.initialv;
else
    initialv=zeros(n,1);
end

if (flag && isfield(opts,'sparsity'))
    sparsity=opts.sparsity;
else
    sparsity=0;
end

RSE(1)=1;

tildev=initialv;
v=initialv;
x=v;


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
    error1=(norm(A*v-b))^2;
end

stopc=0;
iter=0;
theta=2/(iter+2);
times(1)=toc;

while ~stopc
    tic
    iter=iter+1;
    thetaold=theta;
    vold=v;
    theta=2/(iter+2);
     
    sx=max( 0, tildev - 1 ) - max( 0, -tildev - 1);
    x=mu*sx;
    Axb=A*x-b;
    v=tildev-alpha*(A'*Axb);

    tk=1+theta*(thetaold^(-1)-1);
    tildev=tk*v+(1-tk)*vold;


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
%% setting Output
Out.error=RSE;
Out.iter=iter;
Out.times=times;

end