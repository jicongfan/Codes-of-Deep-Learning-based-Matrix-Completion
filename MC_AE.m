function [Xr,AEMC]=MC_AE(X,M,s,options)
% This is the code for the following paper:
% Jicong Fan, Tommy Chow. Deep Learning based Matix Completion.
% Neurocomputing, 2017.
% Written by Jicong Fan, 2017. E-mail: fanj.c.rick@gmail.com
m=size(X,2);
AEMC = myNNsetup([m s(1) m]);
AEMC.activation_func= {options.act_func{1},options.act_func{2}};
AEMC.weight_penalty_L2=options.weight_decay;
AEMC.MC=1;
AEMC.O=M;
disp('Greedy layer-wise training......')
AEMC=AEMC_optimization(AEMC,X);
AEs{1}=AEMC;
Xr0=X.*M+AEMC.a{end}.*~M;
if length(s)>=2
    Z=AEMC.a{2}(:,2:end);
    for i=2:length(s)
        AE = myNNsetup([s(i-1) s(i) s(i-1)]);
        AE.activation_func = {options.act_func{(i-1)*2+1},options.act_func{(i-1)*2+2}};
        AE.weight_penalty_L2=options.weight_decay;% linear output 0.05; nonlinear output 0.001
        AE.MC=0;
        AE=myNN_optimization(AE,Z);
        Z=AE.a{2}(:,2:end);
        AEs{i}=AE;
    end
    SAE=myNNsetup([m s fliplr(s(1:end-1)) m]);
    for i=1:length(s)
        SAE.activation_func{i} = AEs{i}.activation_func{1};
        SAE.activation_func{length(s)*2+1-i} = AEs{i}.activation_func{2};
        SAE.W{i}=AEs{i}.W{1};
        SAE.W{length(s)*2+1-i}=AEs{i}.W{2};
    end
    disp('Fine-tuning......')
    SAE.MC=1;
    SAE.O=M;
    AEMC=AEMC_optimization(SAE,Xr0);
end
Xr=AEMC.a{end};
Xr=X.*M+Xr.*~M;
end

%%
function myNN=AEMC_optimization(myNN,X)
options.Method='cg'; % lbfgs and cg
options.MaxIter=500;
options.MaxFunEvals=2*options.MaxIter;
options.Display='final';
options.optTol=1e-10;
options.progTol=1e-10;
% options.LS_init=0;
% options.LS_interp=0;
% options.LS_type=0;
% options.c2=0.9;
w=[];
myNN = myNNff(myNN, X, X);
for i=1:length(myNN.W)
    w=[w;myNN.W{i}(:)];
end
x=X(myNN.O==0);
y=[x;w];
[y,f,exitflag,output] = minFunc(@fg_AEMC,y,options,myNN);
X(myNN.O==0)=y(1:length(x));
t=length(x)+1;
for i=1:length(myNN.W)
    [a,b]=size(myNN.W{i});
    myNN.W{i}=reshape(y(t:t+a*b-1),a,b);
    t=t+a*b;
end
myNN = myNNff(myNN, X, X);
%
end

%%
function [f,g]=fg_AEMC(y,myNN)
    lx=length(find(myNN.O(:)==0));
    Xt=myNN.a{1}(:,2:end);
    Xt(myNN.O==0)=y(1:lx);
    t=lx+1;
    for i=1:length(myNN.W)
        [a,b]=size(myNN.W{i});
        myNN.W{i}=reshape(y(t:t+a*b-1),a,b);
        t=t+a*b;
    end
    myNN = myNNff(myNN, Xt, Xt);
    myNN = myNNbp(myNN);
    gW=[];
    w=y(lx+1:end);
    for i=1:length(myNN.W)
%         dW=myNN.dW{i}+myNN.weight_penalty_L2*[zeros(size(myNN.W{i},1),1) myNN.W{i}(:,2:end)];
%         gW=[gW;dW(:)];
        gW=[gW;myNN.dW{i}(:)+myNN.weight_penalty_L2*myNN.W{i}(:)];
    end
    f=myNN.loss+0.5*myNN.weight_penalty_L2*sum(w.^2);
    gX=myNN.dX;
    gX=gX(myNN.O==0);
    g=[gX; gW];
end
