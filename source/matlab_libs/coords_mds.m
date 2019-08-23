% Pablo Gainza LPDI STI EPFL 2018-2019
% Compute patch coordinates for MaSIF. 
% This code reads a matlab matrix with a protein shape, decomposes it into radial patches, 
%                       and computes radial coordinates and angular coordinates

function success = coords_mds(paths, params)
    % Computes radial and angular coordinates for 
    % all points in a patch with respect to its center points
    % and saves them to disk.
    % Due to issues with matlab's sparse functions poorly handling updates, this implementation computes 
    % coordinates in batches and then incorporates these batches into a sparse matrix

    % shape instances
    tmp = dir(fullfile(paths.input, '*.mat'));
    names = sort({tmp.name}); clear tmp;
    radius = params.radius;
    success = 0;

    fprintf('Running... \n');
    % loop over the shape instances
    %par
    tStart = tic;

    for idx_shape = 1:length(names)

        % re-assigning structs variables to avoid parfor errors
        paths_ = paths;

        % current shape
        name = names{idx_shape}(1:end - 4);

        % avoid unnecessary computations
        if exist(fullfile(paths_.output, [name, '.mat']), 'file')
            fprintf('[i] shape ''%s'' already processed, skipping\n', name);
            continue;
        end

        % display info
        fprintf('[i] processing shape ''%s'' (%3.0d/%3.0d)... ', name, idx_shape, length(names));
        time_start = tic;

        % load current shape
        tmp = load(fullfile(paths_.input, [name, '.mat']));

        % Check if there are two proteins ('p1' and 'p2) or just one. 
        if isfield(tmp, 'p2')
            list_shapes = [tmp.p1, tmp.p2];
        else
            list_shapes = [tmp.p1];
        end

        all_patch_coord = [];
        list_patch_coord_names = {'p1', 'p2'};

        % Go through each of the two proteins (p1 and p2), or just one.
        for idx_shape2 = 1:length(list_shapes)
            shape = list_shapes(idx_shape2);
            % Compute faces adjacent to each vertex. 
            shape.idxs = compute_vertex_face_ring(shape.TRIV');
            verts = [shape.X, shape.Y, shape.Z];
            face = shape.TRIV;
            % Compute the normals of the shape.
            [~, shape.normalf] = compute_normal(verts, face, 0);

            n = size(shape.X, 1);
            % theta: angular coordinates; rho: radial coordinates.
            patch_theta = sparse(n, n);
            patch_rho = sparse(n, n);
            % Precompute graph structure for Dijkstra. A: the edge weights.
            [G, A] = geodesic_dists_graph(shape);
            % Go through each vertex, from 1 to n.
            vertex_indices = 1:n;

            fprintf('Computing coords for shape %s \n', names{idx_shape});
            fprintf('subshape: %d \n', idx_shape2);
            time_theta = 0.0;
            time_dijkstra1 = 0.0;
            sum_all_dijkstra = 0.0;
            time_mds = 0.0;
            time_dijkstra2 = 0.0;
            time_sparse1 = 0.0;
            time_sparse2 = 0.0;
            rho_col = [];
            rho_row = [];
            rho_val = [];
            theta_col = [];
            theta_row = [];
            theta_val = [];
            % Iterate over all points in the shape, extracting a radial patch around each one. 
            for iii = 1:numel(vertex_indices)
                % Print some stats regularly
                if mod(iii, 500) == 0
                    fprintf('vertex: %d \n', iii);
                    fprintf('Dijkstra time 1: %.2f\n', time_dijkstra1);
                    fprintf('Dijkstra time 2: %.2f\n', time_dijkstra2);
                    fprintf('MDS time : %.2f\n', time_mds);
                    fprintf('Theta time : %.2f\n', time_theta);
                    fprintf('Sparse time 1: %.2f\n', time_sparse1);
                    fprintf('Sparse time 2: %.2f\n', time_sparse2);
                    sum_all_dijkstra = sum_all_dijkstra + time_dijkstra1 + time_dijkstra2;

                    time_theta = 0.0;
                    time_dijkstra1 = 0.0;
                    time_mds = 0.0;
                    time_dijkstra2 = 0.0;
                    tic;
                    % For memory/speed reasons, merge coordinates here.
                    patch_theta_tmp = sparse(theta_row, theta_col, theta_val, n, n);
                    patch_rho_tmp = sparse(rho_row, rho_col, rho_val, n, n);
                    patch_theta = patch_theta + patch_theta_tmp;
                    patch_rho = patch_rho + patch_rho_tmp;
                    rho_col = [];
                    rho_row = [];
                    rho_val = [];
                    theta_col = [];
                    theta_row = [];
                    theta_val = [];
                    fprintf('Merge time: %.2f \n', toc);
                    fprintf('\n');
                end

                vix = vertex_indices(iii);
                % Compute the distance between vix and all neighbors.
                tic;
                % Call weighted Dijkstra from vertex vix to all other nodes in the graph
                dists = distances(G, vix);
                time_dijkstra1 = time_dijkstra1 + toc;
                tic;
                % Neigh: all vertices within radius. 
                neigh = find(dists <= radius);
                % gw2: graph weights between neighbors.
                gw2 = A(neigh, neigh);
                G2 = graph(gw2);
                % Compute the pairwise geodesic (Dijkstra) distances between all vertices in the patch.
                all_pairs_dist = distances(G2);
                time_dijkstra2 = toc + time_dijkstra2;
                tic;
                % use the pairwise distance between all vertices in the patch to scale the patch from 2D manifold space to a plane. 
                [mds_map, e] = cmdscale(all_pairs_dist, 2); %Multidimensional scaling to flatten out the surface
                time_mds = toc + time_mds;
                tic;
                % Compute angular coordinates (theta) with respect to a random direction 
                theta_tmp_tmp = compute_theta(mds_map, vix, neigh, shape); 
                time_theta = toc + time_theta;
                tic;
                [row, col, val] = find(theta_tmp_tmp);
                theta_row = [theta_row; row];
                theta_col = [theta_col; col];
                theta_val = [theta_val; val];
                time_sparse1 = time_sparse1 + toc;
                tic;
                % Set nodes where distance is zero to Matlab's epsilon value.
                dists(dists == 0.0) = eps;
                col = find(dists < radius)';
                row = repmat(vix, numel(col), 1);
                val = dists(col)';
                rho_row = [rho_row; row];
                rho_col = [rho_col; col];
                rho_val = [rho_val; val];
                time_sparse2 = time_sparse2 + toc;
            end

            % These tmp variables store the current batch.
            patch_theta_tmp = sparse(theta_row, theta_col, theta_val, n, n);
            patch_rho_tmp = sparse(rho_row, rho_col, rho_val, n, n);
            patch_theta = patch_theta + patch_theta_tmp;
            patch_rho = patch_rho + patch_rho_tmp;
            fprintf('Total time Dijkstra = %f\n', sum_all_dijkstra);

            fprintf('Finished computing patch coordinates\n');
            % Store the 
            % First come the radial (rho) coordinates and then the angular (theta) coordinates.
            patch_coord = [sparse(patch_rho), sparse(patch_theta)];
            all_patch_coord.(list_patch_coord_names{idx_shape2}) = patch_coord;
        end

        % saving
        if ~exist(paths_.output, 'dir')
            mkdir(paths_.output);
        end

        par_save(fullfile(paths_.output, [name, '.mat']), all_patch_coord);
        fprintf('Saved matrix \n');
        success = 1;

        if toc(tStart) > 250
            exit(0);
        end

        % display info
        fprintf('%2.0fs\n', toc(time_start));
        %catch E
        %    fprintf('exception file %s', name);
        %end

    end

end

function par_save(path, all_patch_coord)
    % Save the polar coordinates.
    save(path, 'all_patch_coord', '-v7.3');
end


function thetas = compute_theta(plane, vix, neighbors, shape)
    % Compute_theta: compute the angles of each vertex with respect to some
    % random direction. Ensure that theta runs clockwise with respect to the
    % normals.

    plane_center_ix = find(neighbors == vix);
    thetas = sparse(numel(shape.X), numel(shape.X));
    % Center the plane so that the origin is at (0,0).
    plane = plane - repmat(plane(plane_center_ix, :), numel(neighbors), 1);

    % Choose one of the neighboring triangles arbitrarily
    tt = shape.idxs{vix}(1);
    normal_tt = shape.normalf(:, tt)';
    tt = shape.TRIV(tt, :);

    neigh_tt = tt(find(tt ~= vix));
    v1ix = neigh_tt(1);
    v2ix = neigh_tt(2);
    v1ix_plane = find(neighbors == v1ix);
    v2ix_plane = find(neighbors == v2ix);

    % Order neighboring triangles clockwise.
    %[sorted, ~] = find_adj_triang_clockwise(shape, tt, vix);
    normal_plane = repmat([0.0, 0.0, 1.0], numel(neighbors), 1);

    % Compute the theta angles for all points in plane.
    % Shoot in a random direction.
    vecs = plane ./ repmat(sqrt(sum(plane.^2, 2)), 1, 2);
    vecs(plane_center_ix, :) = [0, 0];
    ref_vec = vecs(v1ix_plane, :);
    ref_vec = ref_vec / norm(ref_vec);
    ref_vec = repmat(ref_vec, numel(neighbors), 1);
    vecs(:, 3) = 0.0;
    ref_vec(:, 3) = 0.0;

    term1 = sqrt(sum(cross(vecs, ref_vec).^2, 2));
    term1 = atan2d(term1, dot(vecs, ref_vec, 2));
    theta = term1 .* sign(dot(vecs, cross(normal_plane, ref_vec), 2));

    % Compute the sign of the angle between v2ix and v1ix
    v0 = [shape.X(vix), shape.Y(vix), shape.Z(vix)];
    v1 = [shape.X(v1ix), shape.Y(v1ix), shape.Z(v1ix)];
    v2 = [shape.X(v2ix), shape.Y(v2ix), shape.Z(v2ix)];
    v1 = v1 - v0;
    v1 = v1 / norm(v1);
    v2 = v2 - v0;
    v2 = v2 / norm(v2);
    angle_v1_v2 = atan2d(norm(cross(v2, v1)), dot(v2, v1)) .* sign(dot(v2, cross(normal_tt, v1)));

    sign_3d = sign(angle_v1_v2);
    sign_2d = sign(theta(v2ix_plane));

    if sign_3d ~= sign_2d
        theta = -theta;
        %fprintf('Flipping theta\n');
    end

    % Set theta == 0 to epsilon to not confuse it in the sparse matrix.
    theta(theta == 0) = eps;
    thetas(vix, neighbors) = deg2rad(theta);

end
