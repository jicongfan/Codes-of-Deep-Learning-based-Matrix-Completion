function [myNN, L]=myNNtrain(myNN, train_x, train_y, opts, val_x, val_y)
assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;

numbatches = m / batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    disp(['Epoch ' num2str(i) ' ......'])
    tic;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(myNN.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>myNN.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        myNN = myNNff(myNN, batch_x, batch_y);
        myNN = myNNbp(myNN);
        myNN = myNNapplygrads(myNN);
        
        L(n) = myNN.loss;
        
        n = n + 1;
    end
    
    t = toc;
% 
%     if opts.validation == 1
%         loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
%         str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
%     else
%         loss = nneval(nn, loss, train_x, train_y);
%         str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
%     end
%     if ishandle(fhandle)
%         nnupdatefigures(nn, fhandle, loss, opts, i);
%     end
%     obj_func(i,:)=L;   
%     disp(['epoch ' num2str(i) '/' num2str(opts.numepochs) '. Took ' num2str(t) ' seconds' '. Mini-batch mean squared error on training set is ' num2str(mean(L((n-numbatches):(n-1)))) str_perf]);
%     myNN.learning_rate = myNN.learning_rate * myNN.learning_rate_scale;
end
end

