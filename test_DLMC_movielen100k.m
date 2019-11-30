clc
clear all
%%
load('movielen100k.mat');%%% 1000 users on 1700 movies
X=X';
M_org=double(X~=0);
X=X/5;
[nr,nc]=size(X);
missrate=0.5;
for pp=1:1
% miss data
M_t=ones(nr,nc);
% random mask
for i=1:nr
    idx=find(X(i,:)~=0);
    lidx=length(idx);
    temp=randperm(lidx,ceil(lidx*missrate));
    temp=idx(temp);
    M_t(i,temp)=0;
end
%
M=M_t.*M_org;
Xo=X.*M;
Xc=[];
k=0;
%% LRMC
k=k+1;
tic
[Xr{k},E]=MC_IALM(Xo,M);
T(k)=toc;
%% DLMC
k=k+1;
disp('SAEMC......')
s=[50 10];
options.act_func={'tanh_opt','sigm','tanh_opt','sigm'};%
options.weight_decay=0.001;%
tic
[Xr{k},AECF]=MC_AE(Xo,M,s,options);
T(k)=toc;
%%
for i=1:length(Xr)
    MM=(~M).*M_org;
    E=(Xr{i}-X).*MM;
    NMAE(pp,i)=sum(abs(E(:)))/sum(MM(:));
end
%
end



