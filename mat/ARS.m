function [W] = ARS(a,sigma, N, nDelta, nTop, W0, policy, doRollout) 


W = W0;
policy_mean = zeros(size(W0,1),1);
policy_std = ones(size(W0,1),1);
totalT = 0;

for e = 1:N
    
    deltas = randn(size(W,1), size(W,2),nDelta).*sigma;
    Wr = cat(3, W+deltas, W-deltas);
    
    Rs = zeros(1, size(Wr,3));
    Xmus = zeros(size(W0,2), size(Wr,3));
    Xvars = ones(size(W0,2), size(Wr,3));
    
    for pii = 1:size(Wr,3)
        Wc = Wr(:,:,pii);
        policy = @(x)(Wc*((x - policy_mean)./policy_std));
        [R,X] = doRollout(policy); % n_delta*2 parrallel rollouts
        Rs(pii) = R;
        Xmus(:,pii) = mean(X,2);
        Xvars(:,pii) = var(X,0,2);
    end
    
%     % update mean and std ---------------------------------------------------------
%     for mi = 1:size(Wr,3)
%         policy_mean = (policy_mean*totalT + Xmus(:,mi))./(totalT + rolloutTs(mi));
%     end 
%     
%     for si = 1:size(Wr,3)
%         if any(Xvars(:,si) < var_thresh)
%            continue 
%         end
%         cur_var = policy_std.^2;
%         new_var = Xvars(:,si);
%         
%         updated_var = (cur_var*totalT + new_var*rolloutTs(si))./(totalT + rolloutTs(mi));
%         policy_std = sqrt(updated_var);
%     end
    % update mean and std ---------------------------------------------------------
    
    pR = Rs(1:nDelta);
    mR = Rs(nDelta+1:end);
    
    tR = max(pR, mR);
    
    [sR, sI] = sort(tR,'descend');
    Rdiff = (pR - mR);
    ss = a/(size(deltas,1)*std(Rs) + 1e-6);
    
    
    sortedRdiff = Rdiff(sI(1:nTop));
    sortedDeltas = deltas(:,:,sI(1:nTop)); 
    step = zeros(size(Wr,1), size(Wr,2)); 
    
    for j = 1:nTop
       step = step + sortedRdiff(j).*sortedDeltas(:,:,j);
    end
       
    W = W + ss*step;
    

end