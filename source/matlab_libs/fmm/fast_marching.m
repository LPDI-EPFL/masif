function [shape,D]=fast_marching(source_center_number,shape,kind_of_start_back_tracking_source,f_dns)

plot_results=0;
if strcmp(kind_of_start_back_tracking_source,'center')%%% pattern center
    xsource=mean(shape.centers.X(source_center_number));
    ysource=mean(shape.centers.Y(source_center_number));
    zsource=mean(shape.centers.Z(source_center_number));
    
    all_norms=sqrt(shape.X.^2+shape.Y.^2+shape.Z.^2);
    inner_product=[xsource ysource zsource]*[shape.X shape.Y shape.Z]'/norm([xsource ysource zsource])./all_norms';
    [closest_v_source v_source]=min(abs(inner_product-1));
else
    v_source=source_center_number; 
end

source = repmat(Inf, [size(shape.X) 1]);
source(v_source)=0;

D = fastmarchmex('march', f_dns, double(source));
D(D>=9999999) = Inf;

if nargout==1,
    shape.D = D;
end

