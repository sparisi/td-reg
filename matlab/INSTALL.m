disp('Installing directories...')

mips_path = mfilename('fullpath');
mips_path = mips_path(1:end-7); % Remove 'INSTALL' from the path
addpath(genpath(mips_path))

disp('Done!')
