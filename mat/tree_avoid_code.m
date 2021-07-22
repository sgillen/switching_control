% Katie says: getting rid of "hitting walls are death" will make finding
% the linear policy easier.


a = 3;
b = 7;
L = 5; 

xvals = [0:.05:10];
xmin = min(xvals); xmax = max(xvals);
Xmid = .5*(xmin+xmax);
dx = xvals(2)-xvals(1);
nx = length(xvals);

yvals = [-2:.05:0.5];
ymin = min(yvals); ymax = max(yvals);
dy = yvals(2)-yvals(1);
ny = length(yvals);

[X,Y] = meshgrid(xvals,yvals);
deadzone = [a b]; % cannot be at or between these values, in x, when y=0

dt = 0.01;
uset = 0.05*[-L:L];   % list of possible actions

% perform value iteration, for optimal policy, and optimal action
V = 0*X;
%fi = find(X==0); % left side no-go zone
%V(fi) = -1000;
%fi = find(X==10); % right side no-go zone
%V(fi) = -1000;
fi = find((X>=deadzone(1)).*(X<=deadzone(2)).*(Y==0)); % obstacle
V(fi) = -1000 + abs(X(fi)-Xmid);

fi_bad = find(V==-1000); % "dead" states...

V_onestep = V(:);  % one-step cost
V = V(:); % optimal cost
%id = 1:length(X(1,:));
id = 1:length(V);
id_go = zeros(length(id),length(uset));
for n=1:length(uset);
    xgo = X(id) + uset(n);
    xgo = max(xmin,xgo);
    xgo = min(xmax,xgo);
    ygo = Y(id) + dy;
    fi = find(ygo > max(yvals));
    ygo(fi) = yvals(1); % top edge "wraps back" to bottom yvals value
    
    xid = round((xgo - xmin) * (1/dx)) + 1;
    yid = round((ygo - ymin) * (1/dy)) + 1;
    id_go(:,n) = (xid - 1) * ny + yid;
end
P = 0*V; % policy
Pid = 0*V; % where policy sends you, in one step
    
df = 0.99; % discount factor
for n1=1:100 % perform the iteration    
    Vtest = 0*id_go;
    for n2=1:length(uset)
        Vtest(:,n2) = V_onestep(id_go(:,n2)) + df*V(id_go(:,n2)) - uset(n2)^2; % u^2 penalty
        Vtest(:,n2) = max(-1000,Vtest(:,n2));
    end
    [ai,bi] = sort(Vtest,2);
    P = bi(:,end); % optimal action (from uset list)
    V = ai(:,end); % highest value (in Vtest)
    %Pid = 
    %keyboard
end

Pmat = 0*X;
Pmat(:) = P; % optimal policy
figure(1); clf
%plot3(X,Y,Pmat,'.')
surf(X,Y,Pmat)
Vmat = 0*X;
Vmat(:) = V; % value
figure(2); clf
surf(X,Y,Vmat)
axis([xmin xmax ymin ymax -10 10])
