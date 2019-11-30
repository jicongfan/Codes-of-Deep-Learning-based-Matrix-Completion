function myNN=myNNsetup(architecture)
%% parameters
myNN.size=architecture;
myNN.layer=numel(myNN.size);
for i=1:myNN.layer-2
    myNN.activation_func{i}='tanh_opt';% default setting (or 'sigm', 'linear')
end
myNN.activation_func{i+1}='linear';% or 'softmax'; the output_function
myNN.learning_rate=1;
myNN.learning_rate_scale=1;
myNN.weight_penalty_L2=0;
myNN.tied_weights=0;
myNN.sparsity_penalty=0;
myNN.sparsity_target=0.05;
myNN.denoising=0;
myNN.inputZeroMaskedFraction=0; 
myNN.dropout_frac=0.1;
% myNN.output_func='sigm';
% myNN.output_func=myNN.activation_func{end}';
myNN.momentum=0;
myNN.MC=0;
%% initialization of weights
for i=2:myNN.layer
    myNN.W{i-1}=(rand(myNN.size(i),myNN.size(i-1)+1)-0.5)*2*4*sqrt(6/(myNN.size(i)+myNN.size(i - 1)));
    myNN.vW{i-1}=zeros(size(myNN.W{i - 1}));
    myNN.p{i}=zeros(1,myNN.size(i));
end
