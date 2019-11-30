clc
clear all
%% load data
load mnist_uint8;
train_x = double(train_x)/255;
test_x  = double(test_x)/255;
train_y = double(train_y);
test_y  = double(test_y);
%% masking
rand('state',0)
idx=randperm(60000,500);
X=train_x(idx,:);
missrate=0.5;
M=ones(size(X));
for i=1:size(X,1)
    id=randperm(size(M,2),ceil(size(M,2)*missrate));
    M(i,id)=0;
end
Xo=X.*M;
%% LRMC
disp('NNM(IALM)......')
k=k+1;
tic
[Xr{k},E]=MC_IALM(Xo,M);
T(k)=toc;
method_name{k}='NNM(IALM)';
%% DLMC
k=k+1;
s=[200 200];%%%%%%%%%%%%%%%%%%%%%%%%
options.act_func={'tanh_opt','sigm','tanh_opt','sigm'};%,'tanh_opt','sigm','tanh_opt','sigm'};
options.weight_decay=0.001;% 0.001 for tanh-sigm; 0.1(0.05) for tanh-linear
tic
[Xr{k},AECF]=MC_AE(Xo,M,s,options);
T(k)=toc;
method_name{k}='SAEMC';
%% evaluation
for i=1:length(Xr)
    E=Xr{i}-X;
    RMSE(i)=norm(E.*~M,'fro')^2/norm(X.*~M,'fro')^2;
end
%% show with images
y=train_y(idx,:);[temp,y]=max(y');y=y';
c=size(X,2)^0.5;
for i=1:10
    temp=find(y==i);
    z(i)=temp(2);
end
D=zeros((length(Xr)+2)*c,10*c);
for i=1:length(Xr)+2
    if i==1;temp=X;end
    if i==2;temp=Xo;end
    if i>=3;temp=Xr{i-2};end
    for j=1:10
        D((i-1)*c+1:i*c,(j-1)*c+1:j*c)=reshape(temp(z(j),:),c,c)';
    end
end
figure;imshow(D)

