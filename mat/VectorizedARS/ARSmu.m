function [W] = ARSmu(a_sched, sigma_sched, N, n_delta, nTop, xvals)

W = zeros(1,4);
W(3) = 5;
W(4) = .8;


for e = 1:N
    
    a = interp1([1,N], a_sched, e);
    sigma = interp1([1,N], sigma_sched, e);
    
    deltas = randn(n_delta,4)*sigma;
    Wr = [W + deltas; W - deltas];
    [R,X] = doRolloutMu(Wr,xvals);
    pR = R(1:n_delta);
    mR = R(n_delta+1:end);
    
    tR = max(pR, mR);
    
    [sR, sI] = sort(tR,'descend');
    Rdiff = (pR - mR);
    ss = a/(size(deltas,1)*std(R) + 1e-6);
    ss
    step  = Rdiff(sI(1:nTop))'*deltas(sI(1:nTop),:);
    W = W + ss*step;
    
    
end

end
