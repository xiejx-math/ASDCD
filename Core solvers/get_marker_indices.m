function idx = get_marker_indices(x, interval)
%GET_MARKER_INDICES Generate sample indices based on horizontal axis x
% x : horizontal axis (e.g., median CPU time)
% interval: spacing in seconds for placing markers (e.g., 0.05s)
%
% Returns indices in x to be used as 'MarkerIndices' for plotting

if isempty(x) || interval <= 0
    idx = []; return;
end
[~,indexM]=max(x);
xs = 0:interval:x(indexM); % desired sampling locations along time-axis
idx = arrayfun(@(v) find(x >= v, 1, 'first'), xs); % find closest available points
idx = unique(idx); % remove duplicates
end
