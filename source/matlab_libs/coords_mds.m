% Pablo Gainza LPDI STI EPFL 2018-2019
% Compute patch coordinates in matlab. 

function success = coords_mds(paths,params)

% shape instances
tmp   = dir(fullfile(paths.input,'*.mat'));
names = sort({tmp.name}); clear tmp;
radius = params.radius;
success = 0;

fprintf('Running... \n');
% loop over the shape instances
%par
tStart = tic;
for idx_shape = 1:length(names)

    % re-assigning structs variables to avoid parfor errors
    paths_  = paths;
    
    % current shape
    name = names{idx_shape}(1:end-4);
    
    % avoid unnecessary computations
    if exist(fullfile(paths_.output,[name,'.mat']),'file')
        fprintf('[i] shape ''%s'' already processed, skipping\n',name);
        continue;
    end
    
    % display info
    fprintf('[i] processing shape ''%s'' (%3.0d/%3.0d)... ',name,idx_shape,length(names));
    time_start = tic;
    
    % load current shape
    tmp = load(fullfile(paths_.input,[name,'.mat']));
    if isfield(tmp, 'p2')
        list_shapes = [tmp.p1, tmp.p2];
    else
        list_shapes = [tmp.p1];
    end
    all_patch_coord = [];
    list_patch_coord_names = {'p1', 'p2'};


    % Go through each of the two proteins.
    for idx_shape2 = 1 : length(list_shapes)
        shape = list_shapes(idx_shape2);
        shape.idxs = compute_vertex_face_ring(shape.TRIV');
        verts = [shape.X, shape.Y, shape.Z];
        face = shape.TRIV;
        [~, shape.normalf] = compute_normal(verts, face, 0);
        
        n = size(shape.X,1);
        patch_theta = sparse(n,n);
        patch_rho = sparse(n,n);
        % Precompute graph structure for nearest neighbors. 
        [G,A] = geodesic_dists_graph(shape, radius);
        % Go through each vertex.
        vertex_indices = 1:n;
        % Compute all distances using fast marching method. 
        %shape.f_dns = fastmarchmex('init', int32(shape.TRIV-1), double(shape.X(:)), double(shape.Y(:)), double(shape.Z(:)));
        %fprintf('Starting fast marching computation.\n');
        %all_pairs_fmm = zeros(n,n);
        %tic
        %for iii = 1:numel(vertex_indices)
        %    if mod(iii, 500) == 0
        %        fprintf('iii = %d\n',iii); 
        %    end
        %    vi = vertex_indices(iii);
        %    [~,D1] = fast_marching(vi, shape, 'vertex', shape.f_dns); 
        %    all_pairs_fmm(vi, :) = D1;
        %end
        %toc
        %fprintf('Making sparse\n');
        %tic
        %all_pairs_fmm(all_pairs_fmm > params.radius*2) = 0;
        %all_pairs_fmm = sparse(all_pairs_fmm);

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
        for iii = 1:numel(vertex_indices)
            if mod(iii,500) == 0
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
                patch_theta_tmp = sparse(theta_row, theta_col, theta_val, n, n);
                patch_rho_tmp = sparse(rho_row, rho_col, rho_val, n, n);
                patch_theta = patch_theta+ patch_theta_tmp;
                patch_rho = patch_rho+ patch_rho_tmp;
                rho_col = [];
                rho_row = [];
                rho_val = [];
                theta_col = [];
                theta_row = [];
                theta_val = [];
                fprintf('Merge time: %.2f \n',toc);
                fprintf('\n');
            end
            vix = vertex_indices(iii);
            % Compute the distance between vix and all neighbors.
            tic;
            dists = distances(G, vix);
            time_dijkstra1 = time_dijkstra1 + toc;
            tic;
            neigh = find(dists <= radius);
            gw2 = A(neigh,neigh);
            G2 = graph(gw2);
            all_pairs_dist = distances(G2);   
            time_dijkstra2 = toc+time_dijkstra2;
            tic;
            [mds_map,e] = cmdscale(all_pairs_dist, 2);
            time_mds = toc+time_mds;
            tic;
            theta_tmp_tmp = compute_theta(mds_map, vix, neigh, shape);
            time_theta = toc+time_theta;
            tic;
            [row, col, val] = find(theta_tmp_tmp);
            theta_row = [theta_row; row];
            theta_col = [theta_col; col];
            theta_val = [theta_val; val];
            %patch_theta = patch_theta + theta_tmp_tmp;
            time_sparse1 = time_sparse1 + toc;
            tic;
            % Set nodes where distance is zero to epsilon
            dists(dists == 0.0) = eps;
            col = find(dists < radius)';
            row = repmat(vix, numel(col), 1);
            val = dists(col)';
            rho_row = [rho_row; row];
            rho_col = [rho_col; col];
            rho_val = [rho_val; val];
            %patch_rho_tmp = sparse(row, col, val, n, n);
            %patch_rho = patch_rho_tmp+patch_rho;
            time_sparse2 = time_sparse2+toc;
        end          
        patch_theta_tmp = sparse(theta_row, theta_col, theta_val, n, n);
        patch_rho_tmp = sparse(rho_row, rho_col, rho_val, n, n);
        patch_theta = patch_theta+ patch_theta_tmp;
        patch_rho = patch_rho+ patch_rho_tmp;
        fprintf('Total time Dijkstra = %f\n', sum_all_dijkstra);

        % compute the patches
        fprintf('Finished computing patch coordinates\n');
        patch_coord = [sparse(patch_rho),sparse(patch_theta)];
        all_patch_coord.(list_patch_coord_names{idx_shape2}) = patch_coord;
    end
    % saving
    if ~exist(paths_.output,'dir')
        mkdir(paths_.output);
    end
    par_save(fullfile(paths_.output,[name,'.mat']),all_patch_coord);
    fprintf('Saved matrix \n');
    success = 1;

    if toc(tStart)>250
        exit(0);
    end
    % display info
    fprintf('%2.0fs\n',toc(time_start));
    %catch E
    %    fprintf('exception file %s', name);
    %end
    
end

end

function par_save(path,all_patch_coord)
save(path,'all_patch_coord','-v7.3');
end

% Compute_theta: compute the angles of each vertex with respect to some
% random direction. Ensure that theta runs clockwise with respect to the
% normals.
function thetas = compute_theta(plane, vix, neighbors, shape)
    
    plane_center_ix = find(neighbors == vix);
    thetas = sparse(numel(shape.X), numel(shape.X));
    % Center the plane so that the origin is at (0,0). 
    plane = plane-repmat(plane(plane_center_ix,:), numel(neighbors), 1);
    
    % Choose one of the neighboring triangles. 
    tt = shape.idxs{vix}(1);
    normal_tt = shape.normalf(:,tt)';
    tt = shape.TRIV(tt,:);

    neigh_tt = tt(find(tt ~= vix));
    v1ix = neigh_tt(1);
    v2ix = neigh_tt(2);
    v1ix_plane = find(neighbors == v1ix);
    v2ix_plane = find(neighbors == v2ix);

    % Order neighboring triangles clockwise. 
    %[sorted, ~] = find_adj_triang_clockwise(shape, tt, vix);
    normal_plane = repmat([0.0,0.0,1.0], numel(neighbors), 1);
    
    % Compute the theta angles for all points in plane. 
    % Shoot in a random direction. 
    vecs = plane./repmat(sqrt(sum(plane.^2,2)), 1,2);
    vecs(plane_center_ix,:) = [0,0];
    ref_vec = vecs(v1ix_plane,:);
    ref_vec = ref_vec/norm(ref_vec);
    ref_vec = repmat(ref_vec,numel(neighbors), 1);
    vecs(:,3) = 0.0; 
    ref_vec(:,3) = 0.0;
    
    term1 = sqrt(sum(cross(vecs,ref_vec).^2,2));
    term1 = atan2d(term1, dot(vecs,ref_vec,2));
    theta = term1.* sign(dot(vecs,cross(normal_plane,ref_vec),2));
        
    % Compute the sign of the angle between v2ix and v1ix 
    v0 = [shape.X(vix), shape.Y(vix), shape.Z(vix)];
    v1 = [shape.X(v1ix), shape.Y(v1ix), shape.Z(v1ix)];
    v2 = [shape.X(v2ix), shape.Y(v2ix), shape.Z(v2ix)];
    v1 = v1 - v0;
    v1 = v1/norm(v1);
    v2 = v2 - v0;
    v2 = v2/norm(v2);
    angle_v1_v2 = atan2d(norm(cross(v2,v1)),dot(v2,v1)).* sign(dot(v2,cross(normal_tt,v1)));
 
    sign_3d = sign(angle_v1_v2);
    sign_2d = sign(theta(v2ix_plane));
    if sign_3d ~= sign_2d
        theta = -theta;
        %fprintf('Flipping theta\n');
    end
    % Set theta == 0 to epsilon to not confuse it in the sparse matrix. 
    theta(theta == 0) = eps;
    thetas(vix,neighbors) = deg2rad(theta);
    
end


