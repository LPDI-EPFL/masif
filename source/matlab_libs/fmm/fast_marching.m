function [shape, D] = fast_marching(source_center_number, shape, f_dns)
    % Calls the C version of the fast marching algorithm
    % Kimmel, Ron, and James A. Sethian. "Computing geodesic paths on manifolds."
    % Proceedings of the national academy of Sciences 95.15 (1998): 8431-8435.

    v_source = source_center_number;

    source = repmat(Inf, [size(shape.X) 1]);
    source(v_source) = 0;

    D = fastmarchmex('march', f_dns, double(source));
    D(D >= 9999999) = Inf;

    if nargout == 1,
        shape.D = D;
    end

end
