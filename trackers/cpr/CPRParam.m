classdef CPRParam
  % CPR Parameter notes 20170502
  % 
  % CPRLabelTracker (CPRLT) owns the ultimately-used parameter structure 
  % (.sPrm) used in training/tracking. It is partially replicated in 
  % RegressorCascade but this is an impl detail.
  %
  % CPRLabelTracker uses the older/more-complete form of parameters that
  % has redundancies, (many) currently unused fields, etc. These
  % parameters come from the original CPR code. The defaults for these are
  % given in .../cpr/param.example.yaml.
  %
  % For ease-of-use, APT users are shown a reduced/simplified set of
  % parameters, sometimes called "new" parameters. The defaults for these
  % are given in .../cpr/params_apt.yaml.
  %
  % There is a conversion routine (below) for going from new-style
  % parameters to old-style parameters.
  %
  % Forward/maintenance plan:
  % * If user loads an existing project, CPRLT modernizes .sPrm on load.
  % This is done by reading param.example.yaml (which is checked-in and 
  % updated with every pull) and "overlaying" the project contents on top.
  % In this way, defaults provided in param.example.yaml are used for any
  % new parameters. The old-style .sPrm is then converted to new-style 
  % params if/when modification in the UI is desired.
  % * If user creates a new project, default tracking params are taken from 
  % the last new-style params used if avail; otherwise the default
  % old-style params (converted to new).
  
  methods (Static)
    
    function [sOld,trkNFrmsSm,trkNFrmsLg,trkNFrmsNear] = ...
        new2old(sNew,nphyspts,nviews)
      % Convert new-style parameters to old-style parameters. Defaults are
      % used for old fields when appropriate. Some old-style fields that 
      % are currently unnecessary are omitted.
      %
      % The additional return args trkNFrms* are b/c the new-style
      % parameters now store some general tracking-related parameters that
      % are stored on lObj rather than in the legacy CPR params.
      
      assert(strcmp(sNew.ROOT.Track.Type,'cpr'));
      
      sOld = struct();
      sOld.Model.name = '';
      sOld.Model.nfids = nphyspts;
      sOld.Model.d = 2;
      sOld.Model.nviews = nviews;
      
      he = sNew.ROOT.Track.HistEq;
      trkNFrmsSm = sNew.ROOT.Track.NFramesSmall;
      trkNFrmsLg = sNew.ROOT.Track.NFramesLarge;
      trkNFrmsNear = sNew.ROOT.Track.NFramesNeighborhood;
      sOld.PreProc.histeq = he.Use;
      sOld.PreProc.histeqH0NumFrames = he.NSampleH0;
      sOld.PreProc.TargetCrop = sNew.ROOT.Track.MultiTarget.TargetCrop;
      sOld.PreProc.NeighborMask = sNew.ROOT.Track.MultiTarget.NeighborMask;
      sOld.PreProc.channelsFcn = [];
      
      cpr = sNew.ROOT.CPR;
      sOld.Reg.T = cpr.NumMajorIter;
      sOld.Reg.K = cpr.NumMinorIter;
      sOld.Reg.type = 1; 
      sOld.Reg.M = cpr.Ferns.Depth;
      sOld.Reg.R = 0;
      sOld.Reg.loss = 'L2';
      sOld.Reg.prm.thrr = [cpr.Ferns.Threshold.Lo cpr.Ferns.Threshold.Hi];
      sOld.Reg.prm.reg = cpr.Ferns.RegFactor;
      switch cpr.RotCorrection.OrientationType
        case 'fixed'
          sOld.Reg.rotCorrection.use = false;
        case {'arbitrary' 'arbitrary trx-specified'}
          sOld.Reg.rotCorrection.use = true;
        otherwise
          assert(false);
      end
      sOld.Reg.rotCorrection.iPtHead = cpr.RotCorrection.HeadPoint;
      sOld.Reg.rotCorrection.iPtTail = cpr.RotCorrection.TailPoint;
      sOld.Reg.occlPrm.Stot = 1; 
%       sOld.Reg.occlPrm.nrows = 3;
%       sOld.Reg.occlPrm.ncols = 3;
%       sOld.Reg.occlPrm.nzones = 1;
%       sOld.Reg.occlPrm.th = 0.5;
      
      sOld.Ftr.type = cpr.Feature.Type;
      sOld.Ftr.metatype = cpr.Feature.Metatype;
      sOld.Ftr.F = cpr.Feature.NGenerate;
      sOld.Ftr.nChn = 1;
      sOld.Ftr.radius = cpr.Feature.Radius;
      sOld.Ftr.abratio = cpr.Feature.ABRatio;
      sOld.Ftr.nsample_std = cpr.Feature.Nsample_std;
      sOld.Ftr.nsample_cor = cpr.Feature.Nsample_cor;
      sOld.Ftr.neighbors = [];
      
      sOld.TrainInit.Naug = cpr.Replicates.NrepTrain;
      sOld.TrainInit.augrotate = []; % obsolete
      sOld.TrainInit.usetrxorientation = ...
        strcmp(cpr.RotCorrection.OrientationType,'arbitrary trx-specified');
      sOld.TrainInit.doptjitter = cpr.Replicates.DoPtJitter;
      sOld.TrainInit.ptjitterfac = cpr.Replicates.PtJitterFac;
      sOld.TrainInit.doboxjitter = cpr.Replicates.DoBBoxJitter;
      sOld.TrainInit.augjitterfac = cpr.Replicates.AugJitterFac;
      sOld.TrainInit.augUseFF = cpr.Replicates.AugUseFF;
      sOld.TrainInit.iPt = [];
      
      sOld.TestInit.Nrep = cpr.Replicates.NrepTrack;
      sOld.TestInit.augrotate = []; % obsolete
      sOld.TestInit.usetrxorientation = sOld.TrainInit.usetrxorientation;
      sOld.TestInit.doptjitter = cpr.Replicates.DoPtJitter;
      sOld.TestInit.ptjitterfac = cpr.Replicates.PtJitterFac;
      sOld.TestInit.doboxjitter = cpr.Replicates.DoBBoxJitter;
      sOld.TestInit.augjitterfac = cpr.Replicates.AugJitterFac;
      sOld.TestInit.augUseFF = cpr.Replicates.AugUseFF;
      sOld.TestInit.movChunkSize = sNew.ROOT.Track.ChunkSize;
      
      sOld.Prune.prune = 1;
      sOld.Prune.usemaxdensity = 1;
      sOld.Prune.maxdensity_sigma = cpr.Prune.MaxDensitySigma; 
    end
    
    function [sNew,npts,nviews] = old2new(sOld,lObj)
      % Convert old-style CPR parameters to APT-style parameters.
      %
      % lObj: Labeler instance. Need this b/c the new parameters include
      % general tracking-related parameters that are set in lObj.
      
      npts = sOld.Model.nfids;
      assert(sOld.Model.d==2);
      nviews = sOld.Model.nviews;
      
      sNew = struct();
      sNew.ROOT.Track.Type = 'cpr';
      sNew.ROOT.Track.HistEq.Use = sOld.PreProc.histeq;
      sNew.ROOT.Track.HistEq.NSampleH0 = sOld.PreProc.histeqH0NumFrames;
      sNew.ROOT.Track.MultiTarget.TargetCrop = sOld.PreProc.TargetCrop;
      sNew.ROOT.Track.MultiTarget.NeighborMask = sOld.PreProc.NeighborMask;
      sNew.ROOT.Track.ChunkSize = sOld.TestInit.movChunkSize;
      sNew.ROOT.Track.NFramesSmall = lObj.trackNFramesSmall;
      sNew.ROOT.Track.NFramesLarge = lObj.trackNFramesLarge;
      sNew.ROOT.Track.NFramesNeighborhood = lObj.trackNFramesNear;
      assert(isempty(sOld.PreProc.channelsFcn));
      
      sNew.ROOT.CPR.NumMajorIter = sOld.Reg.T;
      sNew.ROOT.CPR.NumMinorIter = sOld.Reg.K;
      assert(sOld.Reg.type==1);
      sNew.ROOT.CPR.Ferns.Depth = sOld.Reg.M;
      assert(sOld.Reg.R==0);
      assert(strcmp(sOld.Reg.loss,'L2'));
      sNew.ROOT.CPR.Ferns.Threshold.Lo = sOld.Reg.prm.thrr(1);
      sNew.ROOT.CPR.Ferns.Threshold.Hi = sOld.Reg.prm.thrr(2);
      sNew.ROOT.CPR.Ferns.RegFactor = sOld.Reg.prm.reg;

      % dups assert below for doc purposes
      assert(sOld.TrainInit.usetrxorientation==sOld.TestInit.usetrxorientation);
      if sOld.Reg.rotCorrection.use
        if sOld.TrainInit.usetrxorientation
          sNew.ROOT.CPR.RotCorrection.OrientationType = 'arbitrary trx-specified'; 
        else
          sNew.ROOT.CPR.RotCorrection.OrientationType = 'arbitrary';        
        end
      else
        sNew.ROOT.CPR.RotCorrection.OrientationType = 'fixed';
      end
      sNew.ROOT.CPR.RotCorrection.HeadPoint = sOld.Reg.rotCorrection.iPtHead;
      sNew.ROOT.CPR.RotCorrection.TailPoint = sOld.Reg.rotCorrection.iPtTail;
      
      assert(sOld.Reg.occlPrm.Stot==1); 
%       sOld.Reg.occlPrm.nrows = 3;
%       sOld.Reg.occlPrm.ncols = 3;
%       sOld.Reg.occlPrm.nzones = 1;
%       sOld.Reg.occlPrm.th = 0.5;
      
      sNew.ROOT.CPR.Feature.Type = sOld.Ftr.type;
      sNew.ROOT.CPR.Feature.Metatype = sOld.Ftr.metatype;
      sNew.ROOT.CPR.Feature.NGenerate = sOld.Ftr.F;
      assert(sOld.Ftr.nChn==1);
      sNew.ROOT.CPR.Feature.Radius = sOld.Ftr.radius;
      sNew.ROOT.CPR.Feature.ABRatio = sOld.Ftr.abratio;
      sNew.ROOT.CPR.Feature.Nsample_std = sOld.Ftr.nsample_std;
      sNew.ROOT.CPR.Feature.Nsample_cor = sOld.Ftr.nsample_cor;
      assert(isempty(sOld.Ftr.neighbors));
      
      sNew.ROOT.CPR.Replicates.NrepTrain = sOld.TrainInit.Naug;
      sNew.ROOT.CPR.Replicates.NrepTrack = sOld.TestInit.Nrep;
      if ~isempty(sOld.TrainInit.augrotate)
        assert(sOld.TrainInit.augrotate==sOld.TestInit.augrotate);
        if sOld.TrainInit.augrotate~=sOld.Reg.rotCorrection.use
          warningNoTrace('CPRParam:rot',...
            'TrainInit.augrotate (%d) differs from Reg.rotCorrection.use (%d). Ignoring value of TrainInit.augrotate.',...
            sOld.TrainInit.augrotate,sOld.Reg.rotCorrection.use);
        end
      end
      assert(sOld.TrainInit.usetrxorientation==sOld.TestInit.usetrxorientation);
      assert(sOld.TrainInit.doptjitter==sOld.TestInit.doptjitter);
      assert(sOld.TrainInit.ptjitterfac==sOld.TestInit.ptjitterfac);
      assert(sOld.TrainInit.doboxjitter==sOld.TestInit.doboxjitter);
      assert(sOld.TrainInit.augjitterfac==sOld.TestInit.augjitterfac);
      assert(sOld.TrainInit.augUseFF==sOld.TestInit.augUseFF);
      %sNew.ROOT.CPR.Replicates.AugRotate = sOld.TrainInit.augrotate;
      
      sNew.ROOT.CPR.Replicates.DoPtJitter = sOld.TrainInit.doptjitter;
      sNew.ROOT.CPR.Replicates.PtJitterFac = sOld.TrainInit.ptjitterfac;
      sNew.ROOT.CPR.Replicates.DoBBoxJitter = sOld.TrainInit.doboxjitter;
      sNew.ROOT.CPR.Replicates.AugJitterFac = sOld.TrainInit.augjitterfac;
      sNew.ROOT.CPR.Replicates.AugUseFF = sOld.TrainInit.augUseFF;
      assert(isempty(sOld.TrainInit.iPt));
      
      assert(sOld.Prune.prune==1);
      assert(sOld.Prune.usemaxdensity==1);
      sNew.ROOT.CPR.Prune.MaxDensitySigma = sOld.Prune.maxdensity_sigma; 
    end
    
  end
end