function [x,Out]=mySDCD(A,b,lambda,tau,opts)

% Stochastic dual coordinate descent for linearly constrained convex 
% optimization
%       min f(x) s.t.  Ax=b
%
% the  initial x=0, hence A'r_0=A'(Ax_0-b)=-A'b
%
%Input: the coefficent matrix A, the vector b and opts
%opts.p: the choice for setting p
%opts.TOL: the stopping rule
%
%.....
%
%Output: the approximate solution x and Out
% Out.error: the relative iterative residual \|Ax_k-b\|/\|Ax_0-b\|
% Out.iter: the total number of iteration
% ....
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
    initialz=opts.initial;
else
    initialz=zeros(n,1);
end

if (flag && isfield(opts,'sparsity'))
    sparsity=opts.sparsity;
else
    sparsity=0;
end

%%
z=initialz;
zold=z;
x=max( 0, z - lambda ) - max( 0, -z - lambda );

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
    error1=(norm(A*x-b))^2;
end


%%%
RSE(1)=1;
normAfro=norm(A,'fro')^2;
ell=ceil(m/tau);
blockAnormfro=zeros(ell,1);
for i=1:ell
    if i==ell
        ps=((i-1)*tau+1):1:m;
    else
        ps=((i-1)*tau+1):1:(i*tau);
    end
    Aps=A(ps,:);
    blockAnormfro(i)=norm(A(ps,:),'fro')^2;
    Aarrs{i}=Aps;
    barrs{i}=b(ps);
end
prob=blockAnormfro/normAfro;
cumsumpro=cumsum(prob);

%%%
stopc=0;
iter=0;
times(1)=toc;
while ~stopc
    tic
    iter=iter+1;

    l=sum(cumsumpro<rand)+1;
    AindexR=Aarrs{l};
    bindexR=barrs{l};
    Axb=AindexR*x-bindexR;
    dk=AindexR'*Axb;
    dk_x_xstar=norm(Axb)^2;
    norm_dk=norm(dk)^2;
    alpha=dk_x_xstar/norm_dk;

    %%% update x
    z=z-alpha*dk;
    x=max( 0, z - lambda ) - max( 0, -z - lambda );

%%% checking the stopping rules
   if strategy
        error1=(norm(x-xstar))^2/normxstar;
        RSE(iter+1)=error1;
        if error1<TOL  || iter>=Max_iter
            stopc=1;
        end
    else
        %error(iter+1)=norm(x-xold)/norm(x);
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

