function [W] = ARSmu(a, sigma, N, n_delta, nTop, xvals)

W = zeros(1,4);
W(3) = 5;
W(4) = -.75;


for e = 1:N
    deltas = randn(n_delta,4)*sigma;
    Wr = [W + deltas; W - deltas];
    [R,X] = doRolloutMu(Wr,xvals);
    pR = R(1:n_delta);
    mR = R(n_delta+1:end);
    
    tR = max(pR, mR);
    
    [sR, sI] = sort(tR,'descend');
    Rdiff = (pR - mR);
    ss = a/(size(deltas,1)*std(R) + 1e-6);
    step  = Rdiff(sI(1:nTop))*deltas(sI(1:nTop),:);
    W = W + ss*step;
    
    
end

end

