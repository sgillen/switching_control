env = TreeClass;

nDelta = 256;
nTop = 256;
alpha = .02;
sigma = .05;
N = 500;

begin = tic;
[W, policy] = EnvARSMu(env, alpha, sigma, nDelta, nTop, N);
fprintf("EPS: %f \n",  N*n_delta/toc(begin));


yvals = [-2:.05:0.5];
xvals = [0:.05:10];
[X,Y] = meshgrid(xvals,yvals);
U = zeros(size(X));

for i = 1:size(X,1)
    for j = 1:size(X,2)
        U(i,j) = policy([X(i,j); Y(i,j)]);
    end
end

surf(X,Y,U);
%policy = @(x)(W'*((x - policy_mean)./policy_std));
[R,xhist,thist] = DoRolloutWithEnv(policy,env); % n_delta*2 parrallel
%plot(xhist(1,:), xhist(3,:));