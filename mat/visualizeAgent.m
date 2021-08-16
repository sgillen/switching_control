function visualizeAgent(agent,env)
    d = env.deadzones;
    N = env.N;
    dt = env.dt;
    g = env.g;
    x_vals = 0:.05:10;
    y_vals = 2+dt*N*g:.05:2;
    
    [X,Y] = meshgrid(x_vals,y_vals);
%     Xi = X(:);
%     Yi = Y(:);
%     
%     for i = 1:length(Xi)
%         U(i) = getAction(agent,[Xi(i);Y(i);g]);
%     end

    for i = 1:length(y_vals)
        for j = 1:length(x_vals)
            u(i,j) = double(cell2mat(getAction(agent, [x_vals(j);y_vals(i)])));
        end
    end
    Umat = u;
%     Umat = 0*X;
%     Umat(:) = U; % optimal policy
    figure(); 
    %plot3(X,Y,Pmat,'.')
    surf(X,Y,Umat)
    hold on
    plot3(d(:,1:2)', [d(:,3) d(:,3)]',[2.60442, 2.60442],'r','LineWidth',1)
    xlabel('x'); ylabel('y'); zlabel('actions')
end

