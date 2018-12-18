function autolayout
% AUTOLAYOUT Automatically arranges figures on the screen to avoid 
% overlapping.

    function F = filterFigures(F)
        for i = 1 : numel(F)
            R(i) = true;
            
            for j = 1 : numel(F(i).Children)
                c = F(i).Children(j);
                if isa(F(i).Children(j),'matlab.ui.control.UIControl')
                    R(i) = false;
                end
            end
        end
        F = F(R);
    end

F = get(0,'Children');
if isempty(F), return, end
F = filterFigures(F);
F = mat2cell(F(end:-1:1),ones(1,length(F)))';
if length(F) < 6
    F = reshape([F cell(1,6-length(F))],3,2)';
elseif length(F) < 8
    F = reshape([F cell(1,8-length(F))],4,2)';
elseif length(F) < 15
    F = reshape([F cell(1,15-length(F))],5,3)';
end

sx = size(F,2);
sy = size(F,1);

X = ones(sy,sx)/max(sx,3);
Y = ones(sy,sx)/max(sy,2);

pos = [0,0,1,0.95];

X = X *(pos(3)-pos(1));
Y = Y *(pos(4)-pos(2));

for x = 1 : sx
    for y = 1 : sy
        if ~isempty(F(y,x))
            F{y,x}.Units = 'normalized';
            F{y,x}.OuterPosition = [ pos(1)+sum(X(y,1:x-1),2) , 1-(pos(2)+sum(Y(1:y,x),1)) , X(y,x), Y(y,x)];
        end
    end
end

end
