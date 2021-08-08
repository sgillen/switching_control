function [R, X] = doRolloutMu(W, xvals)
    N = 50;
    x = randsample(xvals,size(W,1),true)';
    x
    ymax = 2;
    y = -ymax*ones(size(x));
    g = 5;
    dt = .01;
    tol = .01;

    R = zeros(size(W,1),1);
    X = zeros(2, size(W,1), N);
    
   
    deadzone = [3 7];
    umax = 5;
    umin = -umax;
    %uset = 0.05*[-5:5];
    %umax = size(uset, 2) - ceil(size(uset,2)/2);
    %umin = -umax;
    
    for i=1:N
        a = sum(W(:,1:2).*([(x-W(:,3)), (y-W(:,4))]),2);
        a = max(umin, a);
        a = min(umax, a);
        u = a;
        
        x = x + u*dt;
        x = max(0, x);
        x = min(10, x);
        y = y + g*dt;
        
        R = R - (u.^2);
        
        term =(x>=deadzone(1)).*(x<=deadzone(2)).*((-tol < y) & (y < tol));
        R = R -10000*term;
        
        X(:,:,i) = [x,y]';
    end


end

