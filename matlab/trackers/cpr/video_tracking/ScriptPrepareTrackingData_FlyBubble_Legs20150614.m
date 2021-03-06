% set up paths
addpath ..;
addpath ../misc;
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/misc/
addpath /groups/branson/home/bransonk/behavioranalysis/code/Jdetect/Jdetect/filehandling/

inmatfiles = {
  '/groups/branson/bransonlab/projects/FlyBubble/PCRLabelled/withfeet/cx_JRC_SS03500_CsChr_RigD_20150512T092542/labeldata5.mat'
  };

savedir = '/groups/branson/home/bransonk/tracking/code/rcpr/data';
savefile = fullfile(savedir,'FlyBubbleLegTestData_20150614.mat');
%savefile = fullfile(savedir,'M135labeleddata.mat');
moviefilestr = 'movie.ufmf';
trxfilestr = 'fixed_registered_trx.mat';

%% put all the data in one mat file

labels = struct;
labels.pts = [];
labels.ts = [];
labels.expidx = [];
labels.flies = [];
labels.expdirs = {};

for i = 1:numel(inmatfiles),

  expdir = fileparts(inmatfiles{i});
  labels.expdirs{i} = expdir;
  d = load(inmatfiles{i});
  
  [npts,ndims,nframes,nflies] = size(d.labeledpos);
  [ts,flies] = find(reshape(all(all(~isnan(d.labeledpos),1),2),[nframes,nflies]));
  if isempty(ts),
    warning('No videos labeled in %s',inmatfiles{i});
    continue;
  end
  if isempty(labels.pts),
    labels.pts = nan(npts,2,0);
  end

  for j = 1:numel(ts),
    t = ts(j);
    fly = flies(j);
    labels.pts(:,:,end+1) = d.labeledpos(:,:,t,fly);
    labels.ts(end+1) = t;
    labels.flies(end+1) = fly;
    labels.expidx(end+1) = i;
  end
  
end

%% save

labels.moviefilestr = moviefilestr;
labels.trxfilestr = trxfilestr;
save(savefile,'-struct','labels');
