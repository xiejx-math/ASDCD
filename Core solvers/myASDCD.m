function [x,Out]=myASDCD(A,b,lambda,tau,opts)
% Stochastic dual coordinate descent with adaptive heavy ball momentum
% for linearly constrained convex optimization
%    min f(x)   s.t.       Ax=b
%
%Input: the coefficent matrix A, the vector b, the block number tau,
% the regularized parameter lambda and opts
%
%opts.TOL: the stopping rule
%
%.....
%
%Output: the approximate solution x and Out
% Out.error: the relative iterative residual
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
normAfro=sum(sum(A.^2));
ell=ceil(m/tau);
blockAnormfro=zeros(ell,1);
for i=1:ell
    if i==ell
        ps=((i-1)*tau+1):1:m;
    else
        ps=((i-1)*tau+1):1:(i*tau);
    end
    Aps=A(ps,:);
    blockAnormfro(i)=full(sum(sum(A(ps,:).^2)));
    Aarrs{i}=Aps;
    barrs{i}=b(ps);
end
prob=blockAnormfro/normAfro;
cumsumpro=cumsum(prob);
%%
xstar_z1z2=0;
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
    z_zoold=z-zold;
    dk_z_zoold=dk'*z_zoold;
    dk_x_xstar=norm(Axb)^2;
    norm_z_zoold=norm(z_zoold)^2;
    norm_dk=norm(dk)^2;
    xstar_x_z_zold=xstar_z1z2-x'*z_zoold;
    denomfm=norm_dk*norm_z_zoold-dk_z_zoold^2;
    if denomfm<10^(-16)
        if norm_dk<10^(-16)
            alpha=1;
            beta=0;
        else
            alpha=dk_x_xstar/norm_dk;
            beta=0;
        end
    else
        alpha=(dk_x_xstar*norm_z_zoold+dk_z_zoold*xstar_x_z_zold)/denomfm;
        beta=(dk_z_zoold*dk_x_xstar+norm_dk*xstar_x_z_zold)/denomfm;
    end
    xstar_z1z2=-alpha*(Axb'*bindexR)+beta*xstar_z1z2;
    %% update x
    zoold=zold;
    zold=z;
    z=z-alpha*dk+beta*(z-zoold);
    x=max( 0, z - lambda ) - max( 0, -z - lambda );
    %%% checking the stopping rules
    if strategy
        error1=(norm(x-xstar))^2/normxstar;
        RSE(iter+1)=error1;
        if error1<TOL  || iter>=Max_iter
            stopc=1;
        end
    else
        res = A * x - b;
        error1 = (res' * res)/normb;
        %error1=(norm(A*x-b))^2/normb;
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