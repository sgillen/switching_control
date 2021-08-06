function [x_next,y_next,R] = step(x_current,y_current, u, g, dt, deadzones,tol)
% Inputs:
% x_current, y_current: self explanatory duh
% u: control action
% g: dy/dt
% dt: time step duh
% deadzones: n x 3 matrix that stores the xmin, xmax, y values of each tree
% tol: y tolerance for the obtacle, used in ARS but 0 for value iteration
% Outputs:
% x_next,y_next: really?? isn't it obvious
% R: reward
x_next = x_current + u*dt; 
y_next = y_current + g*dt;

term = any(x_next>=deadzones(:,1)).*any(x_next<=deadzones(:,2)).*any(-tol<(y_next-deadzones(:,3))&(y_next-deadzones(:,3))<tol); % determines if there is a collision with any of the trees 
R = -1000*term; % -1000 reward when a callision with tree detected
end

