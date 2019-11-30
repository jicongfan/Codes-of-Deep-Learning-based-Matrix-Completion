function [A,E]=MC_IALM(D,M)
% initialize
[m,n]=size(D);
Y = zeros(m,n);
A = Y;
E=Y;
% Iteration 
iter = 0;
converged = false;
mu=0.1;
r=1.01;
maxiter=1000;
e1=1e-8;
e2=1e-8;
normD=norm(D,'fro');
while ~converged
    iter=iter+1;
   temp=D-E+Y/mu;
   A_new=solve_NuclearNorm(temp,mu);
   E_new=(D-A_new+Y/mu).*~M;
   Y=Y+mu*(D-A_new-E_new);
   stopC1=norm(D-A_new-E_new,'fro')/normD;
   stopC2=max(norm(A_new-A,'fro'),norm(E_new-E,'fro'))/normD;
   isstopC=stopC1<e1&&stopC2<e2;
   if isstopC
        disp('converged')
        break;
   end
   if mod(iter,200)==0||isstopC
        disp(['iteration=' num2str(iter) '/' num2str(maxiter) '  rankX=' num2str(rank(A_new,1e-3*norm(A_new,2))) '  mu=' num2str(mu)])
        disp(['stopC1=' num2str(stopC1) '  stopC2=' num2str(stopC2) '  ......'])
   end  
   mu=mu*r;
   A=A_new;
   E=E_new;
end

