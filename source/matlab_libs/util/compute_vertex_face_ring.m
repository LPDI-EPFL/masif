function ring = compute_vertex_face_ring(face, nverts)

% compute_vertex_face_ring - compute the faces adjacent to each vertex
%
%   ring = compute_vertex_face_ring(face);
%
%   Copyright (c) 2007 Gabriel Peyre

[tmp,face] = check_face_vertex([],face);
nfaces = size(face,2);
if nargin < 2
    nverts = max(face(:));
end
ring{nverts} = [];

for i=1:nfaces
    for k=1:3
        ring{face(k,i)}(end+1) = i;
    end
end



function [vertex,face] = check_face_vertex(vertex,face, options)

% check_face_vertex - check that vertices and faces have the correct size
%
%   [vertex,face] = check_face_vertex(vertex,face);
%
%   Copyright (c) 2007 Gabriel Peyre

vertex = check_size(vertex);
face = check_size(face);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function a = check_size(a)
if isempty(a)
    return;
end
if size(a,1)>size(a,2)
    a = a';
end
if size(a,1)<3 && size(a,2)==3
    a = a';
end
if size(a,1)<=3 && size(a,2)>=3 && sum(abs(a(:,3)))==0
    % for flat triangles
    a = a';
end
if size(a,1)~=3
    error('face or vertex is not of correct size');
end
