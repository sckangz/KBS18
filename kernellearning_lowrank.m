function [result]= kernellearning_lowrank(K,y,alpha,beta,gamma,mu)
% initialization
n=length(unique(y));
maxIter=200;
nn=length(y);
Y1=zeros(nn);
Y2=zeros(nn);
J=eye(nn);
m = size(K,3);
g = ones(1,m)/m;
A = zeros(1,m);
B = ones(1,m);
C = 1;
D = zeros(nn);
lb =zeros(1,m);
% Z=rand(n);
 for j = 1:m
        D = D + g(j)*K(:,:,j);
    end

M =zeros(m);
W =D;
%main function
for x = 1:maxIter
   
    Z=(D+mu*eye(nn))\(mu*J+Y1+D);
%     J=J-diag(diag(J));
    Z(find(Z<0))=0;
    HI=zeros(nn);
    for j = 1:m
        HI = HI + g(j)*K(:,:,j);
    end 
    Dold=D;
    D=(mu*eye(nn)+2*gamma*eye(nn))\(mu*W+Y2-eye(nn)*0.5+Z'-0.5*Z*Z'+2*gamma*HI);
    
    D(find(D<0))=0;
    
    G=Z-Y1/mu;
    [U, X, V] = svd(G, 'econ');
    
    diagX = diag(X);
    svp = length(find(diagX > alpha/mu));
    diagX = max(0,diagX - alpha/mu);
    
    if svp < 0.5 %svp = 0
        svp = 1;
    end
    J= U(:,1:svp)*diag(diagX(1:svp))*V(:,1:svp)';
    J(find(J<0))=0;
    
    H=D-Y2/mu;
  
    [U, S, V] = svd(H, 'econ');
    
    diagS = diag(S);
    svp = length(find(diagS > beta/mu));
    diagS = max(0,diagS - beta/mu);
    
    if svp < 0.5 %svp = 0
        svp = 1;
    end
    W= U(:,1:svp)*diag(diagS(1:svp))*V(:,1:svp)';
    
    W(find(W<0))=0;
    
    for a = 1:m
        for b = 1:m
            M(a,b) = trace( K(:,:,a)*K(:,:,b));
        end
        A(a)= 2*gamma*trace( D*K(:,:,a));
    end
    options = optimoptions('quadprog','Algorithm','interior-point-convex','Display','none');
    g = quadprog( 2*M*gamma,-A,[],[],B,C,lb,[],[],options);
    
    Y1=Y1-mu*(Z-J);
    Y2=Y2+mu*(W-D);
    
    mu=mu*1.1;
    
    if((x>5)&(norm(D-Dold,'fro') < norm(Dold,'fro') * 1e-5))
        break
    end
end

L=(J+J')/2;
% actual_ids = spectral_clustering(L, n);
% result=ClusteringMeasure(actual_ids ,y);
V = spectral_clustering(L, n);
for ij=1:20
ids=litekmeans(V, n,  'Replicates', 1);
[res(ij,:)] = Clustering8Measure( y,ids);
end
result(1,1)=mean(res(:,1));result(1,2)=std(res(:,1));
result(2,1)=mean(res(:,2));result(2,2)=std(res(:,2));
result(3,1)=mean(res(:,3));result(3,2)=std(res(:,3));
result(4,1)=mean(res(:,4));result(4,2)=std(res(:,4));
result(5,1)=mean(res(:,5));result(5,2)=std(res(:,5));
result(6,1)=mean(res(:,6));result(6,2)=std(res(:,6));
result(7,1)=mean(res(:,7));result(7,2)=std(res(:,7));
result(8,1)=mean(res(:,8));result(8,2)=std(res(:,8));

end
function [ V] = spectral_clustering(W, k)

D = diag(1./(eps+sqrt(sum(W, 2))));
W = D * W * D;
[U, s, V] = svd(W);
V = U(:, 1 : k);
V = normr(V);

%ids = kmeans(V, k, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
% ids = kmeans(V, k, 'start','sample','maxiter', 1000,'replicates',100,'EmptyAction','singleton');
end
