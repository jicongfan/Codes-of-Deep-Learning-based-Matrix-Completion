function myNN=myNN_optimization(myNN,X)
options.Method='cg';
options.MaxIter=500;
options.MaxFunEvals=2*options.MaxIter;
options.optTol=1e-10;
options.progTol=1e-10;
options.Display='final';
w=[];
myNN = myNNff(myNN, X, X);
for i=1:length(myNN.W)
    w=[w;myNN.W{i}(:)];
end
[w,f,exitflag,output] = minFunc(@fg_NN,w,options,myNN);
t=1;
for i=1:length(myNN.W)
    [a,b]=size(myNN.W{i});
    myNN.W{i}=reshape(w(t:t+a*b-1),a,b);
    t=t+a*b;
end
myNN = myNNff(myNN, X, X);
%
end
%%
function [f,g]=fg_NN(w,myNN)
    t=1;
    for i=1:length(myNN.W)
        [a,b]=size(myNN.W{i});
        myNN.W{i}=reshape(w(t:t+a*b-1),a,b);
        t=t+a*b;
    end
    Xt=myNN.a{1}(:,2:end);
    myNN = myNNff(myNN, Xt, Xt);
    myNN = myNNbp(myNN);
    gW=[];
    for i=1:length(myNN.W)
%         dW=myNN.dW{i}+myNN.weight_penalty_L2*[zeros(size(myNN.W{i},1),1) myNN.W{i}(:,2:end)];
%         gW=[gW;dW(:)];
        gW=[gW;myNN.dW{i}(:)+myNN.weight_penalty_L2*myNN.W{i}(:)];
    end
    f=myNN.loss+0.5*myNN.weight_penalty_L2*sum(w.^2);
    g=[gW];
end