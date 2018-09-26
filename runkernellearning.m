
warning off
  load('C:\Users\User\Desktop\research\kernelclusteringexp\RMKKM\data\wide3.mat')
  load('C:\Users\User\Desktop\research\kernelclusteringexp\RMKKM\data\wide3_kernel\allkernel.mat')
% load('C:\Users\User\Desktop\research\kernelclusteringexp\RMKKM\data\jaffe_213n_676d_10c_uni.mat')
% load('C:\Users\User\Desktop\research\kernelclusteringexp\RMKKM\data\jaffe_213n_676d_10c_uni_kernel\allkernel.mat')

para1=[1e-3 1 ];
para2=[1  100 ];
para3=[ 1e-4 1e-2];
para4=[  1  20];

for ij=1:length(para1)
alpha=para1(ij);
for iji=1:length(para2)
beta=para2(iji);
for ijj=1:length(para3)
gamma=para3(ijj);
for ji=1:length(para4)
    mu=para4(ji);
fprintf('params%12.6f%12.6f%12.6f%12.6f\n',alpha,beta,gamma,mu)
 [result]= kernellearning_sparse(K,y,alpha,beta,gamma,mu)
% [result]= kernellearning_sparse(K,y,alpha,beta,gamma,mu)
 dlmwrite('wide3s.txt',[alpha beta gamma mu result(1,:) result(2,:) result(3,:) result(4,:) result(5,:) result(6,:) result(7,:) result(8,:) ],'-append','delimiter','\t','newline','pc');
end
end
end
end