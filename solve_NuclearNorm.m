function [Z]=solve_NuclearNorm(L,mu)
if min(size(L))<2000
    [U,sigma,V] = svd(L,'econ');
else
    [U,sigma,V] = lansvd(L,0);
end
    sigma = diag(sigma);
    svp = length(find(sigma>1/mu));
    if svp>=1
        sigma = sigma(1:svp)-1/mu;
    else
        svp = 1;
        sigma = 0;
    end
    Z=U(:,1:svp)*diag(sigma)*V(:,1:svp)';
end