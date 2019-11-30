function myNN=myNNff(myNN,X,Y)%% feed_forward pass
n=size(X,1);% number of samples, one row in X is a sample
X=[ones(n,1) X];% add bias 1
L=myNN.layer;
myNN.a{1}=X;
if L~=length(myNN.activation_func)+1
    error('The number of activation functions is in consistent with the number of layers!')
end
% activation 
for i=2:L-1
    switch myNN.activation_func{i-1}
        case 'sigm'
            myNN.a{i}=sigm(myNN.a{i-1}*myNN.W{i-1}');
        case 'tanh_opt'
            myNN.a{i}=tanh_opt(myNN.a{i-1}*myNN.W{i-1}');
        case 'linear'
            myNN.a{i}=myNN.a{i-1}*myNN.W{i-1}';
    end
    if myNN.sparsity_penalty>0
        myNN.p{i}=0.99*myNN.p{i}+0.01*mean(myNN.a{i},1);
    end
    myNN.a{i}=[ones(n,1) myNN.a{i}];
end
% output
switch myNN.activation_func{end} %% The output function is replaced with the activation function for the output layer
    case 'sigm'
        myNN.a{L}=sigm(myNN.a{L-1}*myNN.W{L-1}');
    case 'tanh_opt'
        myNN.a{L}=tanh_opt(myNN.a{L-1}*myNN.W{L-1}');
    case 'linear'
        myNN.a{L}=myNN.a{L-1}*myNN.W{L-1}';
    case 'softmax'
        myNN.a{L}=myNN.a{L-1}*myNN.W{L-1}';
        myNN.a{L}=exp(bsxfun(@minus,myNN.a{L},max(myNN.a{L},[],2)));
        myNN.a{L}=bsxfun(@rdivide,myNN.a{L},sum(myNN.a{L}, 2)); 
end
% pedictive error and value of loss function

myNN.e=Y-myNN.a{L};
if myNN.MC==1
    myNN.e=myNN.e.*myNN.O;
end
switch myNN.activation_func{end}
    case {'sigm', 'linear','tanh_opt'}
        myNN.loss=1/2*sum(sum(myNN.e.^2))/n;
    case 'softmax'
        myNN.loss=-sum(sum(Y.* log(myNN.a{L})))/n;
end
%
end
        
        
        
        
        
