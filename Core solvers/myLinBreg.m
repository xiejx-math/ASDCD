function [x,Out]=myLinBreg(A,b,alpha,mu,opts)

% linearized Bregman iterative method for solving 
%    min \|x\|_1+(1/2mu)\|x\|^2_2   s.t.       Ax=b
%
% the  initial x=0, hence A'r_0=A'(Ax_0-b)=-A'b
%
%Input: the coefficent matrix A, the vector b and opts
%opts.TOL: the stopping rule
%
%.....
%
% Output: the sparse solution x and Out
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
RSE(1)=1;

stopc=0;
iter=0;
%tau_beta=tau/beta;
times(1)=toc;

while ~stopc
    tic
    iter=iter+1;
    
    sx=max( 0, v - 1 ) - max( 0, -v - 1);
    x=mu*sx;
    Axb=A*x-b;
    v=v-alpha*(A'*Axb);


    if strategy
        error1=(norm(x-xstar))^2/normxstar;
        RSE(iter+1)=error1;
        if error1<TOL  || iter>=Max_iter
            stopc=1;
        end
    else
        %error(iter+1)=norm(x-xold)/norm(x);
        error1=(norm(Axb))^2/normb;
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