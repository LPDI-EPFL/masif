function [ result ] = comptue_matlab_matrix(outmat_base, v1, f1, c1, hb1, hph1, v2, f2, c2, hb2, hph2, masif_opts)
% Compute matlab matrices for ply files and shape complementarity values.

outfilename = sprintf('%s.mat', outmat_base);

rng shuffle;

% Increase faces by one for matlab numbering.
f1 = f1+1;

% Compute a normal for shape complementarity calculations.
[n1,nf1] = compute_normal(v1,f1, 0);
s1.X = v1(:,1);
s1.Y = v1(:,2);
s1.Z = v1(:,3);
s1.charge = c1;
s1.hbond = hb1;
s1.hph = hph1;
s1.TRIV = f1;

% Compute vertex areas and neighbors.
s1 = computeVertexAreas(s1);

% Compute shape index
s1.shape_index = computeShapeIndex(s1);
s1.normalv = n1;
s1.normalf = nf1;
n1 = n1';

if masif_opts.single_chain == 0
    % Increase faces by one for matlab numbering.
    f2 = f2+1;
    s2.X = v2(:,1);
    s2.Y = v2(:,2);
    s2.Z = v2(:,3);
    s2.charge = c2;
    s2.hbond = hb2;
    s2.hph = hph2;
    s2.TRIV = f2;
    [n2,nf2] = compute_normal(v2,f2, 0);
    
    s2 = computeVertexAreas(s2);

    s2.shape_index = computeShapeIndex(s2);
    s2.normalv = n2;
    s2.normalf = nf2;
    n2 = n2';
else
    s2 = 0;
end

% Just save the full proteins now. 
if ~isstruct(s2)
    s1.sc_50 = 0;
    s1.sc_25 = 0;
    if s2 ~= 0
        s2.sc_25 = 0;
        sc.sc_50 = 0;
    end
    save_full_mat_file(outfilename, s1, s2, 0) ;
    result = 0;
    return;
end

% Compute the geodesic distances between all vertices. 
fprintf('Computing fast marching method.\n');
s1.f_dns = fastmarchmex('init', int32(s1.TRIV-1), double(s1.X(:)), double(s1.Y(:)), double(s1.Z(:)));
s2.f_dns = fastmarchmex('init', int32(s2.TRIV-1), double(s2.X(:)), double(s2.Y(:)), double(s2.Z(:)));

% Map the distances betwen all vertices of v1 to v2 and vice versa.
[nearest_neighbor_s2_to_s1, dists_s2_to_s1] = knnsearch(v1, v2);
[nearest_neighbor_s1_to_s2, dists_s1_to_s2] = knnsearch(v2, v1);



outer_radius = masif_opts.sc_radius;
radius = outer_radius;
d_cutoff = masif_opts.sc_interaction_cutoff
w = masif_opts.sc_w
scales = [0:outer_radius/10:outer_radius];

% Minimum shape complementarity
min_sc = -10000;
num_rings = 10;
min_num_pairs = 0;

fprintf('Computing shape complementarity of keypoints and assigning residues to XA patches.\n');
max_shape_comp = -1000.0;
max_shape_comp_index = 1;
matched_pairs = sparse(length(v1), length(v2));
% Find vertices within d_cutoff of v2. 
interface_vertices = find(dists_s1_to_s2 < d_cutoff);
% Compute a matrix of distances between all members of interface_vertices.
pd = pdist2(v1(interface_vertices), v1(interface_vertices));
% Get rid of points? Not for now.
K = ceil(numel(interface_vertices));
% Downsample these vertices using farthest point sampling.
S = fps(K, 1, pd);

% Create a sparse matrix for each of the two proteins that will store the shape complementarity vectors. 
s1.sc_25 = sparse(numel(s1.X), 10);
s2.sc_25 = sparse(numel(s2.X), 10);
s1.sc_50 = sparse(numel(s1.X), 10);
s2.sc_50 = sparse(numel(s2.X), 10);

% Store all pairs of that make it through the threshold.
top_pairs = sparse(numel(s1.X), numel(s2.X));

count_pairs = 0;
% Go through every sampled vertex
for cv1_iiix = 1:length(S)
    cv1_ix = interface_vertices(S(cv1_iiix));
    % Check that v1 is within d_cutoff of a vertex in v2.
    if((dists_s1_to_s2(cv1_ix) < d_cutoff))
        % First s1->s2
        [~,D1] = fast_marching(cv1_ix,s1,'vertex',s1.f_dns);
        neigh_cv1 = find(D1 < radius);
        %Find the point cv2_ix in s2 that is closest to cv1_ix
        cv2_ix = nearest_neighbor_s1_to_s2(cv1_ix);
        [~,D2] = fast_marching(cv2_ix,s2,'vertex',s2.f_dns);
        neigh_cv2 = find(D2 < radius);        
        patch_v1 = v1(neigh_cv1,:);
        patch_v2 = v2(neigh_cv2,:);
        patch_n1 = n1(neigh_cv1, :);
        patch_n2 = n2(neigh_cv2,:);
        [p_nearest_neighbor_s2_to_s1, p_dists_s2_to_s1] = knnsearch(patch_v1, patch_v2);
        [p_nearest_neighbor_s1_to_s2, p_dists_s1_to_s2] = knnsearch(patch_v2, patch_v1);
        neigh_cv1_p = p_nearest_neighbor_s1_to_s2;

        comp1 = dot(patch_n1(:,:), -patch_n2(neigh_cv1_p,:),2);
        comp1 = comp1.*exp(-w* (p_dists_s1_to_s2.^2));
        % Use 10 rings such that each ring has equal weight in shape complementarity
        comp_rings1_25 = zeros(num_rings, 1);
        comp_rings1_50 = zeros(num_rings, 1);
        for ring = 1:num_rings
            scale = scales(ring);
            members = find (D1(neigh_cv1) >= scales(ring) & D1(neigh_cv1) < scales(ring+1));
            comp_rings1_25(ring) = prctile(comp1(members), 25);
            comp_rings1_50(ring) = prctile(comp1(members), 50);
        end
        comp1 = median(comp_rings1_25);
        
        % Now s2->s1
        neigh_cv2_p = p_nearest_neighbor_s2_to_s1;
        comp2 = dot(patch_n2(:,:), -patch_n1(neigh_cv2_p,:),2);
        comp2 = comp2.*exp(-w* (p_dists_s2_to_s1.^2));
        % Use 10 rings such that each ring has equal weight in shape complementarity
        comp_rings2_25 = zeros(num_rings, 1);
        comp_rings2_50 = zeros(num_rings, 1);
        for ring = 1:num_rings
            scale = scales(ring);
            members = find (D2(neigh_cv2) >= scales(ring) & D2(neigh_cv2) < scales(ring+1));
            comp_rings2_25(ring) = prctile(comp2(members), 25);
            comp_rings2_50(ring) = prctile(comp2(members), 50);
            if ring == 1 && sum(isnan(comp_rings2_50(1))) > 0
                comp2(members)
                members
            end
        end
        comp2 = median(comp_rings2_25);
        shape_comp = min([comp1,comp2]);

        if shape_comp > min_sc
            count_pairs = count_pairs +1;
            top_pairs(cv1_ix, cv2_ix) = 1;
            s1.sc_25(cv1_ix,:) = comp_rings1_25(:);
            s1.sc_50(cv1_ix,:) = comp_rings1_50(:);
            s2.sc_25(cv2_ix,:) = comp_rings2_25(:);
            s2.sc_50(cv2_ix,:) = comp_rings2_50(:);

%            outfilename1 = sprintf('%s_%d_%d_1.ply', outply_base, cv1_ix, cv2_ix);
%            outfilename2 = sprintf('%s_%d_%d_2.ply', outply_base, cv1_ix, cv2_ix);
%            save_shape_as_ply(subshape1, outfilename1, subshape1.subkeypoint);
%            save_shape_as_ply(subshape2, outfilename2, subshape2.subkeypoint);
        end
    end
end

fprintf('Counted %d pairs \n', count_pairs);
if count_pairs >= min_num_pairs
    fprintf('Saving to %s\n', outfilename);
    save_full_mat_file(outfilename, s1, s2, top_pairs);
else
    fprintf('Number of pairs did not pass threshold.\n');
end

result = 0;

end
function [ ]  = save_full_mat_file(filename, shape1, shape2, sc_pairs)
    p1.X = shape1.X;
    p1.Y = shape1.Y;
    p1.Z = shape1.Z;
    p1.TRIV = shape1.TRIV;
    p1.charge = shape1.charge';
    p1.hbond = shape1.hbond';
    p1.hphob = shape1.hph';
    p1.shape_index = shape1.shape_index;
    p1.normal = shape1.normalv';
    p1.shape_comp_50 = shape1.sc_50;
    p1.shape_comp_25 = shape1.sc_25;

    if isstruct(shape2)
        p2.X = shape2.X;
        p2.Y = shape2.Y;
        p2.Z = shape2.Z;
        p2.TRIV = shape2.TRIV;
        p2.charge = shape2.charge';
        p2.hbond = shape2.hbond';
        p2.hphob = shape2.hph';
        p2.shape_index = shape2.shape_index;
        p2.normal = shape2.normalv';
        p2.shape_comp_50 = shape2.sc_50;
        p2.shape_comp_25 = shape2.sc_25;

        save(filename,'p1', 'p2', 'sc_pairs', '-v7.3');
    else
        save(filename,'p1',  '-v7.3');
    end
        
end
