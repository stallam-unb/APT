% trackBatch('lObj',lObj,'jsonfile',jsonfile)
% Track multiple videos that do not have to be part of the current APT
% project. 
% This is only the start of this function. Currently, only reading details
% of what to track from a json file is implemented. 
% Example json files are in 
% examples/totrack_example*.json
function trackBatch(varargin)

[lObj,jsonfile,toTrack] = myparse(varargin,'lObj',[],'jsonfile','','toTrack',[]);

assert(~isempty(lObj));
if ~isempty(jsonfile),
  % read what to track from json file
  [toTrack] = parseToTrackJSON(jsonfile,lObj);
end
assert(~isempty(toTrack));

if iscell(toTrack.f0s),
  f0s = cell2mat(toTrack.f0s);
else
  f0s = toTrack.f0s;
end
if iscell(toTrack.f1s),
  f1s = cell2mat(toTrack.f1s);
else
  f1s = toTrack.f1s;
end
if size(toTrack.cropRois,2) > 1,
  cropRois = cell(size(toTrack.cropRois,1),1);
  for i = 1:size(toTrack.cropRois,1),
    cropRois{i} = cat(1,toTrack.cropRois{i,:});
  end
else
  cropRois = toTrack.cropRois;
end
if ~iscell(toTrack.targets) && size(toTrack.movfiles,1) == 1,
  toTrack.targets = {toTrack.targets};
end
    

% call tracker.track to do the real tracking
lObj.tracker.track(toTrack.movfiles,'trxfiles',toTrack.trxfiles,'trkfiles',toTrack.trkfiles,...
  'cropRois',cropRois,'calibrationfiles',toTrack.calibrationfiles,...
  'targets',toTrack.targets,'f0',f0s,'f1',f1s);
