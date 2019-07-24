function [normal, normalf] = compute_normal(vertex, face, flip_normal)

    % compute_normal - compute the normal of a triangulation
    %
    %   [normal,normalf] = compute_normal(vertex,face);
    %
    %   normal(i,:) is the normal at vertex i.
    %   normalf(j,:) is the normal at face j.
    %
    %   Copyright (c) 2004 Gabriel Peyr?

    [vertex, face] = check_face_vertex(vertex, face);

    nface = size(face, 2);
    nvert = size(vertex, 2);
    normal = zeros(3, nvert);

    % unit normals to the faces
    normalf = crossp(vertex(:, face(2, :)) - vertex(:, face(1, :)), ...
        vertex(:, face(3, :)) - vertex(:, face(1, :)));
    d = sqrt(sum(normalf.^2, 1)); d(d < eps) = 1;
    normalf = normalf ./ repmat(d, 3, 1);

    % unit normal to the vertex
    normal = zeros(3, nvert);

    for i = 1:nface
        f = face(:, i);

        for j = 1:3
            normal(:, f(j)) = normal(:, f(j)) + normalf(:, i);
        end

    end

    % normalize
    d = sqrt(sum(normal.^2, 1)); d(d < eps) = 1;
    normal = normal ./ repmat(d, 3, 1);

    % enforce that the normal are outward
    v = vertex - repmat(mean(vertex, 1), 3, 1);
    %fprintf('v: \n')
    %v
    s = sum(v .* normal, 2);

    if sum(s > 0) < sum(s < 0)
        fprintf('Flipping\n')
        % flip
        normal = -normal;
        normalf = -normalf;
    end

    if flip_normal
        normal = -normal;
        normalf = -normalf;
        fprintf('%%%%%%%%%%%%FLIPPING NORMALS!!!!');
    else
        fprintf('Not changing normals.');
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    function z = crossp(x, y)
        % x and y are (m,3) dimensional
        z = x;
        z(1, :) = x(2, :) .* y(3, :) - x(3, :) .* y(2, :);
        z(2, :) = x(3, :) .* y(1, :) - x(1, :) .* y(3, :);
        z(3, :) = x(1, :) .* y(2, :) - x(2, :) .* y(1, :);
