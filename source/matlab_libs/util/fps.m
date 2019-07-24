function [ S ] = fps(K, v1, D)
    %METRICFPS Samples K vertices from V by using farthest point sampling.
    % The farthest point sampling starts with vertex v1 and uses the metric
    % given by matrix D
    % -  K is the number of samples
    % -  v1 is the index of the first vertex in the sample set. (1<=v1<=n)
    % -  D is a n-by-n matrix, such that D(i,j) contains distance between
    %    vertices i and j.
    % Returns
    % -  S is a K-dimensional vector that includes the indeces of the K sample
    %    vertices.

    S = zeros(K, 1);
    S(1) = v1;
    d = D(:, v1);

    for i = 2:K
        [~, m] = max(d);
        S(i) = m(1);
        d = min(D(:, S(i)), d);
    end

end
