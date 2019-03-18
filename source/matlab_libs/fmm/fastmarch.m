% fastmarch    Fast marching algorithm for geodesic distance approximation
%
% Usage:
% 
%  D = fastmarch(TRIV, X, Y, Z, [src], [opt])
%  D = fastmarch(surface, [src], [opt])
%
% Description: 
% 
%  Computes the geodesic distances on a triangulated surfaces using 
%  the fast marching algorithm. The algorithm may operate in two modes:
%  single-source and multiple-source. In the single-source mode, a distance
%  map of every mesh vertex from a single source is computed. The source
%  may be a single point on the mesh, or any other configuration described
%  by an initial set of values per mesh vertex. In the multiple-source
%  mode, a matrix of pair-wise geodesic distances is computed between a
%  specified set of mesh vertices.
%
% Input:  
% 
%  TRIV    - ntx3 triangulation matrix with 1-based indices (as the one
%            returned by the MATLAB function delaunay).
%  X,Y,Z   - vectors with nv vertex coordinates.
%  surface - alternative way to specify the mesh as a struct, having .TRIV,
%            .X, .Y, and .Z as its fields.
%  src     - in the multiple-source mode: (default: src = [1:nv])
%             list of ns mesh vertex indices to be used as sources. 
%            in the single-source mode: (must be specified)
%             an nvx1 list of initial values of the distance function on the mesh
%             (set a vertex to Inf to exclude it from the source set). src
%  opt     - (optional) settings
%             .mode - Mode (default: 'multiple')
%               'multiple' - multiple-source
%               'single'   - single-source
%
% Output:
%
%  D       - In the multiple-source mode: 
%             nsxns matrix of approximate geodesic distances, where D(i,j) is
%             the geodesic distance between the i-th and the j-th point, 
%             whose indices are specified by src(i) and src(j),
%             respectively.
%            In the single-source mode:
%             nvx1 vector of approximated geodesic distances, where D(i) is
%             the geodesic distance from the i-th mesh vertex to the
%             source.
%
% References:
%
% [1] R. Kimmel and J. A. Sethian. "Computing geodesic paths on manifolds",
%     Proc. of National Academy of Sciences, USA, 95(15), p. 8431-8435, 1998.
%
% TOSCA = Toolbox for Surface Comparison and Analysis
% Web: http://tosca.cs.technion.ac.il
% Version: 0.9
%
% (C) Copyright Alex Bronstein, 2005-2007
% (C) Portions copyright Moran Feldman, 2003-2004.
% (C) Portions copyright Ron Kimmel.
% All rights reserved.
%
% License:
%
% ANY ACADEMIC USE OF THIS CODE MUST CITE THE ABOVE REFERENCES. 
% ANY COMMERCIAL USE PROHIBITED. PLEASE CONTACT THE AUTHORS FOR 
% LICENSING TERMS. PROTECTED BY INTERNATIONAL INTELLECTUAL PROPERTY 
% LAWS AND PATENTS PENDING.

function [D,L] = fastmarch(TRIV, X, Y, Z, src, opt)

if nargin < 4,
    surface = TRIV;
    TRIV = surface.TRIV;
    X = surface.X;
    Y = surface.Y;
    Z = surface.Z;
    src = surface.keypoints;
end

mode = 0;
if nargin > 5 & isfield(opt, 'mode'),
    if strcmpi(opt.mode, 'multiple'),
        mode = 0;
    elseif strcmpi(opt.mode, 'single'),
        mode = 1;
    else
        error('Invalid mode. Use either "multiple" or "single".');
    end
end

if nargin == 1 | nargin == 4, 
    if mode == 0,
        src = [1:length(X)];
    else
        error('Source set must be specified in single source mode.');
    end
end

if mode & length(src) ~= length(X(:)),
    error('src must be nvx1 in the single source mode.');
end

% MEX implementation
if ~mode,
    [D] = fastmarchmex(int32(TRIV-1), int32(src(:)-1), double(X(:)), double(Y(:)), double(Z(:)));
else
    idx = find(src==0);
    srclabel = zeros(length(src),1);
    srclabel(idx) = 1:length(idx);
    [D,L] = fastmarch1_mex(int32(TRIV-1), double([src(:); srclabel(:)]), double(X(:)), double(Y(:)), double(Z(:)));
end

D(D>=9999999) = Inf;
