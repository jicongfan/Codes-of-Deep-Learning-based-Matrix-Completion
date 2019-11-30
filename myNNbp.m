function myNN=myNNbp(myNN)
L=myNN.layer;
sparsityError = 0;
switch myNN.activation_func{end}
    case 'sigm'
        d{L}=-myNN.e.*(myNN.a{L}.*(1-myNN.a{L}));
    case 'tanh_opt'
        d{L}=-myNN.e.*(1.7159*2/3 *(1-1/(1.7159)^2*myNN.a{L}.^2));
    case {'softmax','linear'}
        d{L}=-myNN.e;
end
for i=L-1:-1:2
    switch myNN.activation_func{i-1}
        case 'sigm'
            d_act=myNN.a{i}.*(1-myNN.a{i});
        case 'linear'
            d_act=1;
        case 'tanh_opt'
            d_act=1.7159*2/3 *(1-1/(1.7159)^2*myNN.a{i}.^2);
    end
    if myNN.sparsity_penalty>0
        pi = repmat(myNN.p{i}, size(myNN.a{i}, 1), 1);
        sparsityError=[zeros(size(myNN.a{i},1),1) myNN.sparsity_penalty*(-myNN.sparsity_target./ pi + (1 - myNN.sparsity_target) ./ (1 - pi))];
    end
    if i+1==L % in this case in d{n} there is not the bias term to be removed             
        d{i} = (d{i + 1} * myNN.W{i} + sparsityError).* d_act; % Bishop (5.56)
    else % in this case in d{i} the bias term has to be removed
        d{i} = (d{i + 1}(:,2:end) * myNN.W{i} + sparsityError) .* d_act;
    end
    
%     if myNN.dropout_frac==0
%         d{i} = d{i} .* [ones(size(d{i},1),1) myNN.dropOutMask{i}];
%     end
end
%
for i = 1:(L - 1)
    if i+1==L
        myNN.dW{i} = (d{i + 1}' * myNN.a{i}) / size(d{i + 1}, 1);
    else
        myNN.dW{i} = (d{i + 1}(:,2:end)' * myNN.a{i}) / size(d{i + 1}, 1);      
    end
end
if myNN.MC==1;
myNN.dX=myNN.e+d{2}(:,2:end)*myNN.W{1}(:,2:end);
myNN.dX=myNN.dX/size(myNN.a{1},1);
myNN.dX=myNN.dX.*~myNN.O;
end
end