function []=compileObjectnessMex()

%% Compile all mex files needed in the objectness code
fprintf('Compiling mex files...\n');
mex bescores.c
mex integralimage.c
mex selectwindows.c
mex wsscores.c
mex scoreSamplingMex.c
fprintf('Done!\n');


