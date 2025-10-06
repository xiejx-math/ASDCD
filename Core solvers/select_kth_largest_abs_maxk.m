function [value, index] = select_kth_largest_abs_maxk(x, s)
    % Select the (s+1)-th largest absolute value element using maxk
    % Input:
    %   x - input vector
    %   s - parameter, select the (s+1)-th largest absolute value element
    % Output:
    %   value - the actual value of the (s+1)-th largest absolute value element
    %   index - the index of the (s+1)-th largest absolute value element
    
    % Calculate absolute values
    abs_x = abs(x);
    
    % Use maxk to find the top s+1 largest absolute values and their indices
    [~, top_indices] = maxk(abs_x, s+1);
    
    % Select the last one (the (s+1)-th largest)
    index = top_indices(end);
    value = abs_x(index);
end