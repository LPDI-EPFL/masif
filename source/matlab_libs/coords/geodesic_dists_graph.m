function [G, graph_weights] = geodesic_dists_graph(shape)
    % Takes in a mesh and returns its underlying graph
    % and the lengths of the edges

    vertex = [shape.X, shape.Y, shape.Z];
    face = shape.TRIV;

    [vertex, face] = check_face_vertex(vertex, face);

    n = size(vertex, 2);
    m = size(face, 2);

    all_pairs_D = sparse(n, n);

    % associate each edge to a pair of faces
    i = [face(1, :) face(2, :) face(3, :)];
    j = [face(2, :) face(3, :) face(1, :)];
    s = [1:m 1:m 1:m];
    [u, ui, ~] = unique([i' j'], 'rows', 'stable');
    s = s(:, ui);
    i = u(:, 1);
    j = u(:, 2);
    A = sparse(i, j, s, n, n);

    [i, j, s1] = find(A); % direct link
    [i, j, s2] = find(A'); % reverse link

    I = find((s1 > 0) + (s2 > 0) == 2);

    % links edge->faces
    E = [s1(I) s2(I)];
    i = i(I); j = j(I);

    % only directed edges
    I = find(i < j);
    E = E(I, :);
    i = i(I); j = j(I);
    ne = length(i); % number of directed edges

    % normalized edge
    e = vertex(:, j) - vertex(:, i);
    d = sqrt(sum(e.^2, 1));
    graph_weights1 = sparse(i, j, d, n, n);
    graph_weights2 = sparse(j, i, d, n, n);
    graph_weights = graph_weights1 + graph_weights2;
    G = graph(graph_weights);

end
