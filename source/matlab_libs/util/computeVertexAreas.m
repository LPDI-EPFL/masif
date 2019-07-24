function [ shape ] = computeVertexAreas(shape)
    % Computes the area of all vertices in the shape.
    % Stores the areas in shape.Av and stores the faces_by_vertex in
    %   shape.faces_by_vertex   .

    shape.faces_by_vertex = compute_vertex_face_ring(shape.TRIV, length(shape.X));
    vertex_areas = zeros(size(shape.X));

    for v1 = 1:numel(shape.X)
        FACES_V1 = shape.faces_by_vertex{v1};
        mysum = 0;

        for faceix = 1:numel(FACES_V1)
            vertices_in_face = shape.TRIV(FACES_V1(faceix), :);
            area = area_triangle(shape.X(vertices_in_face), ...
                shape.Y(vertices_in_face), shape.Z(vertices_in_face));
            mysum = mysum + area;
        end

        vertex_areas(v1) = mysum / numel(FACES_V1);

    end

    shape.Av = vertex_areas;

    function [area] = area_triangle(x, y, z)
        x = x(:)';
        y = y(:)';
        z = z(:)';
        ons = [1 1 1];
        area = 0.5 * sqrt(det([x; y; ons])^2 + det([y; z; ons])^2 + det([z; x; ons])^2);
    end

end
