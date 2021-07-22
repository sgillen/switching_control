clear all;
close all;

a = .1;
sigma = .1;
nDelta = 512;
nTop = 512;
N = 500;

xvals1 =  [0:.05:5];

W0 = zeros(1,2)
policy = @(W, x)(W*x);
doRollout = @(p)doRollout2D(p,xvals1,50); 

W1 = ARS(a,sigma,N,nDelta,nTop,W0,policy, doRollout);




