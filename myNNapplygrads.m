function myNN=myNNapplygrads(myNN) 
for i=1:myNN.layer-1
    if myNN.weight_penalty_L2==0
        dW=myNN.dW{i};
    else
        dW=myNN.dW{i}+myNN.weight_penalty_L2*[zeros(size(myNN.W{i},1),1) myNN.W{i}(:,2:end)];
    end
    dW=dW*myNN.learning_rate;
    if myNN.momentum>0
        my.vW{i} = myNN.momentum*myNN.vW{i} + dW;
        dW=myNN.vW{i};
    end
    myNN.W{i}=myNN.W{i}-dW;
end
if myNN.MC==1
%    dX=(myNN.a{1}(:,2:end)-myNN.a{end}).*~myNN.O;
   myNN.a{1}(:,2:end)=myNN.a{1}(:,2:end)-myNN.learning_rate*myNN.dX;
end

end
