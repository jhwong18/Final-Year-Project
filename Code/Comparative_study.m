% This codes outputs the error for comparative study between Conditional Gradient vs
% Proposed approach
% Run this code iteratively to obtain multiple comparisons in choice prob


% Input:
% For Conditional Gradient:
% b, g0, B (obtained by building B)
% For Proposed Approach:
% marketshare, CPprob

% Output:
% combined_prob_CG (choice prob from CG across cust types)
% total_error (absolute % error in choice prob between proposed and CG)

% Run line 17 to 21 once to convert to array types
% b = table2array(b);
% g0 = table2array(g0);
% CPprob = table2array(CPprob);
% marketshare = table2array(marketshare);
% cluster = table2array(cluster);

% Building B
[d1 n1] = size(cluster);
cluster_1_final = [];
k1 = 1;
while k1 <= 95
    sum_all_cust_clust1 = zeros(1,95);
    num_rows_sample_clust1 = randsample(d1,100);
    for i = 1:100
        sum_all_cust_clust1 = sum_all_cust_clust1 + cluster(num_rows_sample_clust1(i),:); % for each customer, aggregated good count
    end
    sum_clust1 = sum(sum_all_cust_clust1); % total good count for all goods for 1 customer
    sum_all_cust_clust1 = sum_all_cust_clust1/sum_clust1; % choice probability for 1 customer
    cluster_1_final = [cluster_1_final sum_all_cust_clust1'];
    k1 = k1 + 1;
end
B = cluster_1_final;

% Run Conditional Gradient
[f_t,res] = Researchpaper1(g0, b,B);
S_t = res.S_t;
% Removing Duplicates
S_t = unique(S_t.', 'rows').';
S_t_final = S_t;
for k=1:size(S_t,2)
    if all(S_t(:,k) == g0) == 1
        S_t_final(:,k) = [];
    end
end

% Computing the likelihood for each cust type
likelihood = zeros(1,size(S_t_final,2));
likelihood_index = [];
for j=1:size(S_t_final,2)
    for k=1:size(res.S_t,2)
        if all(res.S_t(:,k) == S_t_final(:,j)) == 1
            likelihood_index = [likelihood_index;k];
        end
    end
    likelihood(1,j) = sum(res.alpha_t(likelihood_index));
    likelihood_index = [];
end
% Computing the choice prob across cust type
combined_prob_CG = S_t_final*likelihood';

% Comparison Plot
x1=1:95;
plot(x1,combined_prob_CG) 
hold on 
plot(x1,CPprob)
title('Comparison of Choice Probabilities between CG vs Proposed')
legend('Conditional Gradient','Proposed Approach')
xlabel('Product Index') 
ylabel('Choice Probability') 

% Computing the error between proposed and CG approaches
total_error = sum(marketshare.*abs(combined_prob_CG - CPprob)./CPprob)
