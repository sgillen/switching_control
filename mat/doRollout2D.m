function [R,X] = doRollout2D(policy,xvals,N)
    x = randsample(xvals,1,true);
    y = -2*ones(size(x));
    dy = .05;
    tol = .01;

    X = zeros(2, N);
    R = 0;
    
    
    deadzone = [3 7];
    uset = 0.05*[-5:5];
    umax = size(uset, 2) - ceil(size(uset,2)/2);
    umin = -umax;
    
    for i=1:N
        a = policy([x;y]);
        a = max(umin, a);
        a = min(umax, a);
        u = a;
        
        x = x + u;
        x = max(0, x);
        x = min(10, x);
        y = y + dy;
        
        R = R - (u.^2);
        
        term =(x>=deadzone(1)).*(x<=deadzone(2)).*(-tol < y < tol);
        R = R -1000*term;
        
        X(:,i) = [x;y];
    end

end

