function [W1,W2,net] = ARSS(a_sched, sigma_sched, N, n_delta, nTop, xvals)
    W1 = zeros(1,4); W2 = zeros(1,4);
    W1(3) = 5; W2(3) = 5;
    W1(4) = .8; W2(4) = .8;
    
    net = fitcnet(zeros(1,2),zeros(1)); 
    
    
for e = 1:N
    
    a = interp1([1,N], a_sched, e);
    sigma = interp1([1,N], sigma_sched, e);
    
    deltas = randn(n_delta,4)*sigma;
    
    Wr1 = [W1 + deltas; W1 - deltas];
    Wr2 = [W2 + deltas; W2 - deltas];
    
    xsamples = randsample(xvals, n_delta*2,true)';
    ysamples = .8*ones(n_delta*2,1);
    netX = [xsamples, ysamples];
    netY = net.predict(netX);
    
    Wr = netY.*Wr1 + ~netY.*Wr2;
    
    [R,X] = doRolloutMu(Wr,xvals);
    pR = R(1:n_delta);
    mR = R(n_delta+1:end);
    
    tR = max(pR, mR);
    
    [sR, sI] = sort(tR,'descend');
    Rdiff = (pR - mR);
    ss = a/(size(deltas,1)*std(R) + 1e-6);
    step  = Rdiff(sI(1:nTop))'*deltas(sI(1:nTop),:);
    W1 = W1 + ss*step;
    W2 = W2 + ss*step;
    
    net = fitcnet(

    
end
    
end

