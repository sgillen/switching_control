function [W] = ARSTree(a, sigma, N, n_delta, nTop, xvals)
% Compute variations of the policy W
% Do rollouts with the variants of W
% Change W towards variant Ws with high reward. 


W = zeros(1,2);

for e = 1:N
    deltas = randn(n_delta,2)*sigma;
    Wr = [W + deltas; W - deltas];
    [R,X] = doRolloutTree(Wr,xvals); % n_delta*2 parrallel rollouts
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

