function compile_compute3Dfrom2D()

% get the directory where this file lives
% figure out where the root of the Ohayon code is
thisScriptFileName=mfilename('fullpath');
thisScriptDirName=fileparts(thisScriptFileName);

% just put the executable in the did with the build script
% exeDirName=thisScriptDirName;
exeDirName = ['/groups/branson/bransonlab/mayank/stephen_copy/apt_cache/compiled_' datestr(now,'YYYYmmDD')];

mcc('-o','compute3Dfrom2D_compiled', ...
    '-m', ...
    '-d',fullfile(exeDirName), ...
    '-I','/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling',...
    '-I','/groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc',...
    '-I','/groups/branson/bransonlab/projects/flyHeadTracking/code/',...
    '-I','/groups/branson/home/bransonk/tracking/code/Ctrax/matlab/netlab',...
    '-I','/groups/branson/bransonlab/mayank/APT/matlab/user/orthocam',... % 20191009 update paths to matlab code
    '-I','/groups/branson/bransonlab/mayank/APT/matlab/misc',...
    '-a','/groups/branson/bransonlab/mayank/APT/matlab/user/orthocam/OrthoCam.m',...
    '-a','/groups/branson/bransonlab/mayank/APT/matlab/user/orthocam/OrthoCamCalPair.m',...
    '-v', ...
    '-R','-singleCompThread',...
    fullfile(thisScriptDirName,'compute3Dfrom2D_KB.m'));

