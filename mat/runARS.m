clear all;
close all;

a = .1;
sigma = .1;
n_delta = 512;
nTop = 512;
N = 500;

xvals1 =  [0:.05:5];
W1 = ARS(a,sigma,N,n_delta,nTop,xvals1);


Rt = 0;
Neval = 100;
for e = 1:Neval
    [R,Xt] = doRollout(W1,xvals1);
    Rt = Rt+R;
end

W1
avgR = Rt/Neval
avgR/(Neval*-1000)


%% 

xvals2 =  [0:.05:10];
W2 = ARS(a,sigma,N,n_delta,nTop,xvals2);


Rt = 0;
Neval = 100;
for e = 1:Neval
    [R,Xt] = doRollout(W2,xvals2);
    Rt = Rt+R;
end

W2
avgR = Rt/Neval
avgR/(Neval*-1000)
%% 
yvals = [-2:.05:0.5];
xvals = [0:.05:10];
[X,Y] = meshgrid(xvals,yvals);

U1 = (W1(1)*(X-5) + W1(2)*(Y+.75));
figure(1);
surf(X,Y,U1);
figure(2);
plot(squeeze(Xt(1,1,:)))

U2 = (W2(1)*(X-5) + W2(2)*(Y+.75));
figure(3);
surf(X,Y,U2);
figure(4);
plot(squeeze(Xt(1,1,:)))

W1
W2

figure(5);
U1(X >= 5) = 0;
U2(X < 5) = 0;
Uc = U1 + U2;
surf(X,Y,Uc);



