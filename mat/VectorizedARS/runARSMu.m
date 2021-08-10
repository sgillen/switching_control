clear all;
close all;



a = [1,.01];
sigma = [1,.01];
n_delta = 256;
nTop = 256;
N = 50000;
Neval = 100;

xvals0 =  [0:.05:10];
W0 = ARSmu(a,sigma,N,n_delta,nTop,xvals0);


Rt = 0;
for e = 1:Neval
    [R,Xt] = doRolloutMu(W0,xvals0);
    Rt = Rt+R;
end

W0
W0R = Rt/Neval

%%
xvals1 =  [0:.05:5];
W1 = ARSmu(a,sigma,N,n_delta,nTop,xvals1);


Rt = 0;
for e = 1:Neval
    [R,Xt] = doRolloutMu(W1,xvals1);
    Rt = Rt+R;
end

W1
W1R = Rt/Neval


%% 

xvals2 =  [5:.05:10];
W2 = ARSmu(a,sigma,N,n_delta,nTop,xvals2);


Rt = 0;
for e = 1:Neval
    [R,Xt] = doRolloutMu(W2,xvals2);
    Rt = Rt+R;
end

W2
W2R = Rt/Neval

%%

Rt = 0;
for e = 1:Neval
    [R,Xt] = doRolloutSplit(W1,W2,xvals0);
    Rt = Rt+R;
end

CR = Rt/Neval

%% 
yvals = [-2:.05:0.5];
xvals = [0:.05:10];
[X,Y] = meshgrid(xvals,yvals);


U0 = (W0(1)*(X-W0(3)) + W0(2)*(Y-W0(4)));
figure();
surf(X,Y,U0);
title('single policy')

U1 = (W1(1)*(X-W1(3)) + W1(2)*(Y-W1(4)));
figure();
surf(X,Y,U1);
title('dual policy left')


U2 = (W2(1)*(X-W2(3)) + W2(2)*(Y-W2(4)));
figure();
surf(X,Y,U2);
title('dual policy left')



figure();
U1(X >= 5) = 0;
U2(X < 5) = 0;
Uc = U1 + U2;
surf(X,Y,Uc);
title('dual policy combined')


figure()
surf(X,Y,U0);
hold on
surf(X,Y,Uc);

%% 

Nnn = 1000;
Xnn = zeros(size(Xt,1), Nnn);
Ynn = zeros(1,Nnn);
for i = 1:Nnn
    x = randsample(xvals0,size(W1,1),true)';
    xval = [x x];
    [R1,Xt1] = doRolloutMu(W1, xval);
    [R2,Xt2] = doRolloutMu(W2, xval);
    
    Ynn(:,i) = R1 > R2;
    if R1 > R2
        Xnn(:,i) = Xt1(:,:,1);
    else
        Xnn(:,i) = Xt2(:,:,1);
    end
end

net = fitcnet(Xnn', Ynn); 
Xtest = zeros(100,2);
Xtest(:,1) = [0:.1:9.9]; 
Xtest(:,2) = -1.95;
figure()
plot(Xtest, net.predict(Xtest));


