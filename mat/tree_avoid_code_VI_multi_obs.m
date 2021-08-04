a = 4;
b = 6;
L = 5; 

xvals = [0:.05:10];
xmin = min(xvals); xmax = max(xvals);
Xmid = .5*(xmin+xmax);
dx = xvals(2)-xvals(1);
nx = length(xvals);

yvals = -2.5:.05:0.5;
ymin = min(yvals); ymax = max(yvals);
dy = yvals(2)-yvals(1);
g = .05;
ny = length(yvals);

[X,Y] = meshgrid(xvals,yvals);
% all obstacles 2 units wide, change a and b for changing the obstacles
deadzone1 = [a b]; % obstacle 1 
y_dead1 = -.5;
deadzone2 = [a b]-1.2; % obstacle 2
y_dead2 = 0;
deadzone3 = [a b]+1.2; % obstacle 3
y_dead3 = 0;

dt = 1;
uset = 0.05*[-L:L];   % list of possible actions

% perform value iteration, for optimal policy, and optimal action
V = 0*X;
%fi = find(X==0); % left side no-go zone
%V(fi) = -1000;
%fi = find(X==10); % right side no-go zone
%V(fi) = -1000;
fi1 = find((X>=deadzone1(1)).*(X<=deadzone1(2)).*(Y==y_dead1)); % obstacle 1
% fi2 = find((X>=deadzone2(1)).*(X<=deadzone2(2)).*(Y==y_dead2)); % obstacle 2
% fi3 = find((X>=deadzone3(1)).*(X<=deadzone3(2)).*(Y==y_dead3)); % obstacle 3
V(fi1) = -1000 + abs(X(fi1)-Xmid);
% V(fi2) = -1000 + abs(X(fi2)-Xmid);
% V(fi3) = -1000 + abs(X(fi3)-Xmid);

fi_bad = find(V==-1000); % "dead" states...

V_onestep = V(:);  % one-step cost 
V = V(:); % optimal cost
%%%%%%%%%%%%%%
id = 1:length(V);
id_go = zeros(length(id),length(uset));
for n=1:length(uset)
    xgo = X(id) + uset(n)*dt; 
    xgo = max(xmin,xgo);
    xgo = min(xmax,xgo);
    ygo = Y(id) + g*dt;
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
    Q = 0*id_go;
    for n2=1:length(uset)
        Q(:,n2) = V_onestep(id_go(:,n2))- (uset(n2))^2 + df*V(id_go(:,n2)); % u^2 penalty
        Q(:,n2) = max(-1000,Q(:,n2));
    end
    [ai,bi] = sort(Q,2);
    P = bi(:,end); % optimal action (from uset list)
    V = ai(:,end); % highest value (in Vtest)
    %Pid = 
    %keyboard
end

Pmat = 0*X;
Pmat(:) = P; % optimal policy
figure(); 
%plot3(X,Y,Pmat,'.')
surf(X,Y,Pmat)
Vmat = 0*X;
Vmat(:) = V; % value
figure(); 
surf(X,Y,Vmat)
axis([xmin xmax ymin ymax -10 10])
