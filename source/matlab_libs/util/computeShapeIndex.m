function [ shape_index ] = computeShapeIndex( shape )
%COMPUTE_SHAPE_INDEX Computes the shape index of the shape.
%   Detailed explanation goes here
%% First compute curvature on full shape.
faces = shape.TRIV;
vertices = [shape.X, shape.Y, shape.Z];
options.curvature_smoothing = 1 ;
[Umin,Umax,Cmin,Cmax,Cmean,Cgauss,normals] = compute_curvature(...
            transpose(vertices),transpose(faces), options);
shape_index = 2/pi * atan((Cmax + Cmin)./( Cmax-Cmin ));
end
