% Plot the results
clear all;
close all;

x_bounds = [];
set_legend = 0;
outlier_frac = 0.2;

% Not worth putting.
% file_name = 'Shekel-n0.10-M12-pareto(k=2)-c50-0515-000331.mat';
% file_name = 'Hartmann3-n0.05-M4-halfnormal-c50-0513-173842.mat'; set_legend = 1;
% file_name = 'Hartmann3-n0.05-M4-halfnormal-c50-0514-193354.mat'; set_legend = 1;
% file_name = 'Hartmann3-n0.05-M4-halfnormal-c50-0514-223623.mat'; set_legend = 1;
% file_name = 'auton/Shekel-n0.10-M12-pareto(k=2)-c50-0515-000331.mat';
% file_name = 'auton/Hartmann6-n0.05-M8-exponential-c50-0515-000037.mat';
% file_name = 'auton/Hartmann3-n0.05-M16-halfnormal-c20-0514-235249.mat';
% file_name = 'Shekel-n0.10-M12-exponential-c50-0515-222352.mat';
% file_name = 'coma/Branin-n0.05-M8-uniform-c50-0515-222728.mat';
% file_name = 'coma/Branin-6-n0.20-M30-uniform-c20-0517-154518.mat';
% file_name = 'Borehole-n1.00-M8-exponential-c50-0515-222533.mat';
% file_name = 'Borehole-n10.00-M8-uniform-c50-0516-135517.mat';
% file_name = 'Borehole-n10.00-M4-uniform-c50-0516-115953.mat';

% REpeat
% file_name = 'coma/Hartmann6-n0.05-M12-exponential-c40-0515-221606.mat';



% No diff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file_name = 'CurrinExp-n0.01-M4-uniform-c50-0516-213842.mat'; set_legend = 0;
% file_name = 'Park1-n0.20-M4-uniform-c50-0516-221036.mat';
% file_name = 'Park2-n0.10-M10-halfnormal-c25-0517-003440.mat'; set_legend = 0;
% file_name = 'coma/Park2-16-n0.50-M35-halfnormal-c20-0517-131957.mat'; set_legend = 1;


% May be %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file_name = 'Branin-n0.01-M4-halfnormal-c40-0517-013539.mat'; set_legend=0;

% Final %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% file_name = 'Hartmann3-n0.05-M8-exponential-c30-0515-122906.mat'; set_legend = 0;
% file_name = 'Hartmann6-n0.05-M12-exponential-c30-0515-135136.mat'; set_legend = 0;
% file_name = 'coma/Hartmann12-n0.50-M15-pareto(k=1)-c40-0515-222323.mat'; set_legend = 0;
file_name = '/Users/rajat/Dropbox/MFTree/examples/results/Letters-XGB-p1-d5-0513-024825.mat';
% file_name = 'coma/CurrinExp-14-n0.50-M35-pareto(k=3)-c20-0517-132018.mat';

% error('stop!');

plot_order = {'mf_gp_ucb', 'gp_ucb', 'gp_ei', ...
              'mf_gp_ucb_finite', 'mf_sko'};
plot_colours = {'k', 'r', 'm', 'b', 'k', [255 128 0]/255};
plot_markers = {'.', '.', '.', '.', '.'};
plot_line_markers = {'--', '--', '--', '--', '-'};
plot_legends = plot_order;
% plot_legends{7} = 'asyHUCB';
% plot_legends{2} = 'synHUCB';
x_label = 'Capital $(\Lambda)$';
y_label = 'Simple Regret';
% plot_colours = [];
% plot_markers = [];

% Synchronous only
% plot_order = {'synRAND', 'synBUCB', 'synUCBPE', 'synTS'};
% plot_legends = plot_order;
% plot_legends{2} = 'synHUCB';

% % Asynchronous only
% plot_order = {'asyRAND', 'asyUCB', ...
%               'asyBUCB', 'asyEI', ...
%               'asyHTS', 'asyTS'};
% plot_legends = plot_order;
% plot_legends{3} = 'synHUCB';
% plot_colours = {'k', [255 128 0]/255, 'r', 'g', 'c', 'b'};


% Cifar experiment.
set_legend=1;
% file_name = 'auton/Cifar10-6-M4-real-time-c6000-0517-013019.mat';
% plot_order = {'synBUCB', 'synTS', 'asyRAND', 'asyBUCB', 'asyEI' 'asyTS'};
% plot_legends = plot_order;
% plot_legends{1} = 'synHUCB';
% plot_legends{4} = 'asyHUCB';
% plot_colours = {'r', 'b', 'k', 'r', 'g', 'b'};
% plot_markers = {'x', 's', '+', '>', 'v', 'o'};
% plot_line_markers = {'--', '--', '-', '-', '-', '-'};
outlier_frac = 0.0;
% x_label = 'Time $(s)$';
% y_label = 'Validation Error';
% 
% file_path = sprintf('./final_results/%s', file_name)
results = load(file_name)

%gen_curves_no_markers_duplicate(plot_order, plot_legends, results, x_label,  y_label, ...set_legend, x_bounds, [], outlier_frac, plot_colours, plot_markers, plot_line_markers);

plot_curves(plot_order, plot_legends, results, x_label, y_label,outlier_frac <= 0.0)
