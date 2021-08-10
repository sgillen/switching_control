clear all;
close all;

a = [1,.01];
sigma = [1,.01];
n_delta = 512;
nTop = 256;
N = 10000;
Neval = 100;

xvals0 =  [0:.05:10];
W0 = ARSS(a,sigma,N,n_delta,nTop,xvals0);


Rt = 0;
for e = 1:Neval
    [R,Xt] = doRolloutMu(W0,xvals0);
    Rt = Rt+R;
end

W0
W0R = Rt/Neval