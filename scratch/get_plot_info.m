function [plot_mean, plot_std] = get_plot_info(meth_curr_opt_vals, meth_costs, ...
                                   grid_pts, true_opt_val, outlier_frac)

  % Prelims
  num_experiments = numel(meth_curr_opt_vals);
  outlier_low_idx = round(outlier_frac * num_experiments);
  outlier_high_idx = num_experiments - round(outlier_frac * num_experiments);
  inlier_idxs = outlier_low_idx:outlier_high_idx;
  num_inlier_experiments = numel(inlier_idxs);
  num_grid_pts = numel(grid_pts);

  % Compute values for each experiment at the grid points
  grid_vals = zeros(num_experiments, num_grid_pts);
  for exp_iter = 1:num_experiments
    cum_costs = cumsum(meth_costs{exp_iter});
    num_exp_iters = numel(cum_costs);
    add_err = 1e-9 * (1:num_exp_iters)';
    grid_vals(exp_iter, :) = interp1(cum_costs, ...
                               meth_curr_opt_vals{exp_iter}, grid_pts);
  end
  
  % If true_opt_val is finite, then we plot out regret.
  if true_opt_val < inf & true_opt_val > -inf
    display('I am here!')
    grid_vals = true_opt_val - grid_vals;
  end
  % Sort results column wise
  sorted_grid_vals = sort(grid_vals, 1);
  inlier_grid_vals = sorted_grid_vals(inlier_idxs(2:length(inlier_idxs)), :);

  [plot_mean, plot_std] = compute_array_mean_and_std(inlier_grid_vals);

end


function [arr_means, arr_stds] = compute_array_mean_and_std(arr)
  num_cols = size(arr, 2);
  arr_means = zeros(1, num_cols);
  arr_stds = zeros(1, num_cols);
  for i = 1:num_cols
    [curr_mean, curr_std] = compute_column_mean_and_std(arr(:, i));
    arr_means(i) = curr_mean;
    arr_stds(i) = curr_std;
  end
end


function [col_mean, col_std] = compute_column_mean_and_std(col_vals)
  finite_col_vals = col_vals(isfinite(col_vals));
  num_finite_col_vals = numel(finite_col_vals);
  finite_frac = num_finite_col_vals/numel(col_vals);
  if finite_frac >= 0.4
    col_mean = mean(finite_col_vals);
    col_std = std(finite_col_vals)/sqrt(num_finite_col_vals);
  else
    col_mean = nan;
    col_std = nan;
  end
end
