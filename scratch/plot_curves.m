function plot_curves(plot_order, plot_legends, results, x_label_name, y_label_name, ...
                     set_legend, x_bounds, plot_func, outlier_frac)

  % Set global parameters for plotting
%   plot_colours = {'m', 'g', 'r', 'k', 'c', 'y', 'm', [255 128 0]/255, ...
%     [76, 0, 153]/253, [102 102 0]/255};
%   plot_markers = {'o', '+', '*', 'x', 's', 'd', '^', 'p', '>', 'v'};
  plot_colours = {'g', 'r', 'k',  [255 128 0]/255, 'b', 'b', 'm', 'c', ...
    [76, 0, 153]/253, [102 102 0]/255};
  plot_markers = {'s', 'x', '^', 'p', 'o', '*', '>', 'v'};
  MS = 8;
  LW = 4;
  FS = 30;
  LEG_FS = 20;
  AXIS_FS = 25;
  NUM_GRID_PTS = 100;
  NUM_ERR_BARS = 10;
  close all;

  % Preliminaries
  max_capital = sum(results.query_costs{2,1});
  [num_methods, ~] = size(results.query_costs);
  num_plot_methods = numel(plot_order);
  methods_as_cell = get_methods_as_cell(results.methods);

  % Determine bounds and plot function
  if ~exist('plot_func', 'var') | isempty(plot_func)
    plot_func = @semilogy;
  end
  if ~exist('x_bounds', 'var') | isempty(x_bounds)
    x_bounds = [0, max_capital];
  end
  if ~exist('outlier_frac', 'var') | isempty(outlier_frac)
    outlier_frac = 0.0;
  end
  if ~exist('set_legend', 'var') | isempty(set_legend)
    set_legend = 1;
  end
  
  % Determine the grid for plotting and error bars
  if isequal(plot_func, @semilogx) | isequal(plot_func, @loglog)
    grid_pts = logspace(log10(x_bounds(1)), log10(x_bounds(2)), NUM_GRID_PTS);
  else
    grid_pts = linspace(x_bounds(1), x_bounds(2), NUM_GRID_PTS);
  end
  err_bar_idx_half_gap = 0.5 * NUM_GRID_PTS/NUM_ERR_BARS;
  err_bar_idxs = round(linspace(err_bar_idx_half_gap, ...
                                NUM_GRID_PTS - err_bar_idx_half_gap, NUM_ERR_BARS));

  % Now get the plot means and stds
  plot_means = zeros(num_methods, NUM_GRID_PTS);
  plot_stds = zeros(num_methods, NUM_GRID_PTS);
  for i = 1:num_methods
    meth_curr_opt_vals = results.true_curr_opt_vals(i, :);
    meth_costs = results.query_costs(i, :);
    [meth_plot_mean, meth_plot_std] = get_plot_info(meth_curr_opt_vals, meth_costs, ...
                                                  grid_pts, ...
                                                  results.true_opt_val, outlier_frac);
    plot_means(i,:) = meth_plot_mean;
    plot_stds(i,:) = meth_plot_std;
  end

  % Obtain error bar coordinates
  err_bar_pts = grid_pts(:, err_bar_idxs);
  err_bar_means = plot_means(:, err_bar_idxs);
  err_bar_stds = plot_stds(:, err_bar_idxs);

  % Now perform the plots.
  % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  figure;
  % Error bar points
  for m = 1:num_plot_methods
    meth_idx = get_meth_idx(plot_order{m}, methods_as_cell);
    plot_func(err_bar_pts, err_bar_means(meth_idx, :), plot_markers{m}, ...
      'Color', plot_colours{m}, 'MarkerSize', MS, 'LineWidth', LW);
    hold on,
  end
  % Curves
  for m = 1:num_plot_methods
    meth_idx = get_meth_idx(plot_order{m}, methods_as_cell);
    plot_func(grid_pts, plot_means(meth_idx, :), 'Color', plot_colours{m}, ...
              'LineWidth', LW);
  end
  % Error bars and plot markers again.
  for m = 1:num_plot_methods
    meth_idx = get_meth_idx(plot_order{m}, methods_as_cell);
    errorbar(err_bar_pts, err_bar_means(meth_idx, :), err_bar_stds(meth_idx, :), ...
             plot_markers{m}, ...
             'Color', plot_colours{m}, 'MarkerSize', MS, 'LineWidth', LW);
    plot_func(err_bar_pts, err_bar_means(meth_idx, :), plot_markers{m}, ...
      'Color', plot_colours{m}, 'MarkerSize', MS, 'LineWidth', LW);
    hold on,
  end

  % Limits
  min_plot_val = min(min(plot_means));
  max_plot_val = max(max(plot_means));
  plot_range = max_plot_val - min_plot_val;
  plot_range, min_plot_val, max_plot_val,
  ylim([min_plot_val - 0.02 * plot_range, max_plot_val + 0.1 * plot_range])
  x_range = x_bounds(2) - x_bounds(1);
  xlim([x_bounds(1), x_bounds(2) + 0.01*plot_range])
  % Legends
  if set_legend
    legend(plot_legends, 'FontSize', LEG_FS);
  end
  % Title
  title_str = sprintf('%s, $\\quad p=%d,\\;$ $\\quad d=%d\\;$', ...
    results.experiment_name, results.fidel_dim, results.domain_dim);
  title(title_str, 'Interpreter', 'Latex', 'FontSize', FS);
  xlabel(x_label_name, 'Interpreter', 'Latex', 'FontSize', FS);
  ylabel(y_label_name, 'Interpreter', 'Latex', 'FontSize', FS);
  set(gca, 'FontSize', AXIS_FS);
  %set(gca, 'Position', [0.125 0.11 0.87 0.79], 'units', 'normalized');
   %set(gca, 'Position', [0.125 0.11 0.87 0.79], 'units', 'normalized');
  


end


function idx = get_meth_idx(method, methods_as_cell)
  for i = 1:numel(methods_as_cell)
    if strcmp(methods_as_cell{i}, method)
      idx = i;
      return
    end
  end
  idx = 0;
end


function methods_as_cell = get_methods_as_cell(method_str)
  num_methods = size(method_str, 1);
  methods_as_cell = cell(num_methods, 1);
  for i = 1:num_methods
    methods_as_cell{i} = strtrim(method_str(i,:));
  end
end

