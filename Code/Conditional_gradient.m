function [f_t,res] = Conditional_gradient(g0, b,B)
% opts.Tmax  % max number of iterations
% opts.TOL   % tolerance for convergence (FW gap stopping criterion)
% opts.verbose % whether to print info
% opts.pureFW % set to non-zero to use FW
% if opts.pureFW is undefined or set to 0, then this runs away-steps FW
opts.Tmax  = 2000; % max number of iteration
opts.TOL   = 1e-8; % tolerance for convergence
opts.verbose = true;

% init:
% alpha_t will be a weight vector so that x_t = S_t * alpha_t
[d n] = size(B);
alpha_t = zeros(95,1);
alpha_t(1) = 1;
g_t = g0;  % this is tracking A\alpha

% tracking results:
I_active = find(alpha_t > 0);
fvalues = [];
gap_values = [];
number_away = 0;
number_drop = 0; % counting drop steps (max stepsize for away step)
k = 0;
S = [g0];
S1 = [g0];
G = [g0];

pureFW = isfield(opts, 'pureFW') && opts.pureFW;
if pureFW
    fprintf('running plain FW, for at most %d iterations\n', opts.Tmax);
else  % use away step
    fprintf('running FW with away steps, for at most %d iterations\n', opts.Tmax);
end

% optimization:
it = 1;
while (it <= opts.Tmax && k<94)
    it = it + 1;
    % cost function:
    cost_fun =@(x) sum((x-b).^2)/2;
    grad1 =@(x) (x-b);
    f_t = cost_fun(g_t);
    % gradient
    grad = (g_t - b);
    % towards direction search:
    [id_FW,s_FW]   = LMO(grad,B); % the linear minimization oracle, returning an atom
    size(s_FW)
    size(g_t)
    d_FW     = s_FW - g_t;
    
    % duality gap:
    gap = - d_FW' * grad;
    
    fvalues(it-1) = f_t;
    gap_values(it-1) = gap;
    
    %   if gap < opts.TOL
    %     fprintf('end of FW: reach small duality gap (gap=%g)\n', gap);
    %     break
    %   end
    % away direction search:
%     size(grad)
%     size(B)
%     size(I_active)
    [id_A, v_A]   = away_step(grad, B, I_active);
    d_A    = g_t - v_A;
    alpha_max = alpha_t(id_A);
    
    % construct direction (between towards and away):
    
    if pureFW || - gap <= d_A' * grad
        is_aw = false;
        d = d_FW;
        max_step = 1;
    else % away step
        is_aw = true;
        number_away = number_away+1;
        d = d_A;
        %max_step = 5/(it+2);
        max_step = alpha_max / (1 - alpha_max);
    end
    
    % line search:
    %step = - (grad' * d) / ( d' * d );
    
    % simpler predefined stepsize
    step = 2 / (it+2);
    %step = max(0, min(step, max_step ));
    
    
    if opts.verbose
        fprintf('it = %d -  f = %g - gap=%g - stepsize=%g\n', it, f_t, gap, step);
    end
    
    if step < -eps
        % not a descent direction???
        fprintf('ERROR -- not descent direction???')
        keyboard
    end
    
    % doing steps and updating active set:
    
    if is_aw
        %fprintf('  AWAY step from index %d\n',id_A);
        % away step:
        alpha_t = (1+step)*alpha_t; % note that inactive should stay at 0;
        if abs(step - max_step) < 10*eps
            % drop step:
            number_drop = number_drop+1;
            alpha_t(id_A) = 0;
            % remove from active set
        else
            alpha_t(id_A) = alpha_t(id_A) - step;
        end
    else
        % FW step:
        k = k + 1;
        S = [S grad1(g_t)];
        S1 = [S1 s_FW];
        [alpha,alpha_t] = alpha_step1(S,k,b);
        g_t = G_step(S,alpha);
        %       alpha_final = [alpha_final alpha_t];
        alpha_t = (1-step)*alpha_t;
        alpha_t(id_FW) = alpha_t(id_FW) + step;
        
        I_active = find(alpha_t > 0); %TODO: could speed up by checking if id_FW is new
        
        % exceptional case: stepsize of 1, this collapses the active set!
        if step > 1-eps
            I_active = [id_FW];
        end
    end
    
end


%LMO returns the index of, and the atom of max inner product with the negative gradient
    function [id, s] = LMO(grad,B)
        ips = grad' * B;
        [~,id] = min(ips);
        s = B(:,id);
    end


% returns the id of the active atom with the worst value w.r.t. the
% gradient
    function [id, v] = away_step(grad, B, I_active)
        ips = grad' * B(:,I_active);
        [~,id] = max(ips);
        id = I_active(id(1));
        v = B(:,id);
    end

    function [alpha,alpha_t] = alpha_step1(S,k,b)
        alpha0 = 0.5*ones(k+1,1);
        alpha1 = zeros(95,1) + [alpha0;zeros(95-(k+1),1)];
        fun = @(alpha)ones(1,95)*((sum((S.*alpha')'))'-b);
        A = [];
        C = [];
        Aeq = ones(1,k+1);
        Beq = [1];
        lb = 0.001*ones(k+1,1);
        ub = 0.999*ones(k+1,1);
        x = fmincon(fun,alpha0,A,C,Aeq,Beq,lb,ub);
        alpha = x;
        alpha_t = zeros(95,1)+ [x;zeros(95-(k+1),1)];
    end



    function G_k = G_step(S,alpha_t)
        G_k = S*alpha_t;
    end

res.primal = fvalues;
res.gap = gap_values;
res.number_away = number_away;
res.number_drop = number_drop;
res.S_t = S1;
res.alpha_t = alpha_t;
res.x_t = S1(:,end);

end

% g0 = table2array(g0);
% b = table2array(b);
% B = table2array(B1);
% [f_t,res] = Researchpaper1(g0, b,B);
% res