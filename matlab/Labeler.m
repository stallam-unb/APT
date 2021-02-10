classdef Labeler < handle
% Bransonlab Animal Video Labeler/Tracker

  properties (Constant,Hidden)
    VERSION = '3.1';
    DEFAULT_LBLFILENAME = '%s.lbl';
    DEFAULT_CFG_FILENAME = 'config.default.yaml';
    MAX_MOVIENAME_LENGTH = 80;
    
    % non-config props
  	% KB 20190214 - replaced trackDLParams, preProcParams with trackParams
    SAVEPROPS = { ...
      'VERSION' 'projname' ...
      'movieReadPreLoadMovies' ...
      'movieFilesAll' 'movieInfoAll' 'trxFilesAll' 'projMacros'...
      'movieFilesAllGT' 'movieInfoAllGT' ...
      'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' ...
      'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT' ...
      'trxFilesAllGT' ...
      'cropIsCropMode' ...
      'viewCalibrationData' 'viewCalProjWide' ...
      'viewCalibrationDataGT' ...
      'labeledpos' 'labeledpostag' 'labeledposTS' 'labeledposMarked' 'labeledpos2' ...
      'labeledposGT' 'labeledpostagGT' 'labeledposTSGT' 'labeledpos2GT'...
      'currMovie' 'currFrame' 'currTarget' 'currTracker' ...
      'gtIsGTMode' 'gtSuggMFTable' 'gtTblRes' ...
      'labelTemplate' ...
      'trackModeIdx' 'trackDLBackEnd' ...
      'suspScore' 'suspSelectedMFT' 'suspComputeFcn' ...
      'trackParams' 'preProcH0' 'preProcSaveData' ...
      'xvResults' 'xvResultsTS' ...
      'fgEmpiricalPDF'...
      'projectHasTrx'...
      'skeletonEdges','showSkeleton','flipLandmarkMatches'...
      'trkResIDs' 'trkRes' 'trkResGT' 'trkResViz'};
    SAVEPROPS_LPOS = {...
      'labeledpos' 'nan'
      'labeledposGT' 'nan'
      'labeledpos2' 'nan'
      'labeledpos2GT' 'nan'
      'labeledposTS' 'ts'
      'labeledposTSGT' 'ts'
      'labeledpostag' 'log'
      'labeledposMarked' 'log'
      'labeledpostagGT' 'log'};
%     SAVEPROPS_GTCLASSIFY = { ... % these props are resaved into stripped lbls pre gt-classify
%       'movieFilesAllGT'
%       'movieInfoAllGT'
%       'movieFilesAllGTCropInfo'
%       'movieFilesAllGTHistEqLUT'
%       'trxFilesAllGT'
%       'viewCalibrationDataGT'
%       'labeledposGT'
%       'labeledpostagGT'
%       'labeledposTSGT'
%       'labeledpos2GT'};
    
    SAVEBUTNOTLOADPROPS = { ...
       'VERSION' 'currFrame' 'currMovie' 'currTarget'};     
     
    DLCONFIGINFOURL = 'https://github.com/kristinbranson/APT/wiki/Deep-Neural-Network-Tracking'; 
    
    NEIGHBORING_FRAME_MAXRADIUS = 10;
    DEFAULT_RAW_LABEL_FILENAME = 'label_file.lbl';
  end
  properties (Hidden)
    % Don't compute as a Constant prop, requires APT path initialization
    % before loading class
    NEIGHBORING_FRAME_OFFSETS;    
  end
  properties (Constant,Hidden)
    PROPS_GTSHARED = struct('reg',...
      struct('MFA','movieFilesAll',...
             'MFAF','movieFilesAllFull',...
             'MFAHL','movieFilesAllHaveLbls',...
             'MFACI','movieFilesAllCropInfo',...
             'MFALUT','movieFilesAllHistEqLUT',...
             'MIA','movieInfoAll',...
             'TFA','trxFilesAll',...
             'TFAF','trxFilesAllFull',...
             'LPOS','labeledpos',...
             'LPOSTS','labeledposTS',...
             'LPOSTAG','labeledpostag',...
             'LPOS2','labeledpos2',...
             'VCD','viewCalibrationData',...
             'TRKRES','trkRes'),...
             'gt',...
      struct('MFA','movieFilesAllGT',...
             'MFAF','movieFilesAllGTFull',...
             'MFAHL','movieFilesAllGTHaveLbls',...
             'MFACI','movieFilesAllGTCropInfo',...
             'MFALUT','movieFilesAllGTHistEqLUT',...
             'MIA','movieInfoAllGT',...
             'TFA','trxFilesAllGT',...
             'TFAF','trxFilesAllGTFull',...
             'LPOS','labeledposGT',...
             'LPOSTS','labeledposTSGT',...
             'LPOSTAG','labeledpostagGT',...
             'LPOS2','labeledpos2GT',...
             'VCD','viewCalibrationDataGT',...
             'TRKRES','trkResGT'));
  end
  
  events
    newProject
    projLoaded
    newMovie
    startAddMovie
    finishAddMovie
    startSetMovie
      
    % This event is thrown immediately before .currMovie is updated (if 
    % necessary). Listeners should not rely on the value of .currMovie at
    % event time. currMovie will be subsequently updated (in the usual way) 
    % if necessary. 
    %
    % The EventData for this event is a MoviesRemappedEventData which
    % provides details on the old->new movie idx mapping.
    movieRemoved
    
    % EventData is a MoviesRemappedEventData
    moviesReordered
  end
      
  
  %% Project
  properties (SetObservable)
    projname              % init: PN
    projFSInfo;           % filesystem info
    projTempDir;          % temp dir name to save the raw label file
  end
  properties (SetAccess=private)
    projMacros = struct(); % scalar struct, filesys macros. init: PN
  end
  properties
    % TODO: rename this to "initialConfig" or similar. This is now a
    % "configuration" not "preferences". For the most part this does not 
    % dup Labeler properties, b/c many of the configuration 
    %
    % scalar struct containing noncore (typically cosmetic) prefs
    % TODO: rename to projConfig or similar
    %
    % Notes 20160818: Configuration properties
    % projPrefs captures minor cosmetic/navigation configuration
    % parameters. These are all settable, but currently there is no fixed
    % policy about how the UI updates in response. Depending on the
    % (sub)property, changes may be reflected in the UI "immediately" or on
    % next use; changes may only be reflected after a new movie; or chnages
    % may never be reflected (for now). Changes will always be saved to
    % project files however.
    %
    % A small subset of more important configuration (legacy) params are
    % Dependent and forward (both set and get) to a subprop of projPrefs.
    % This gives the user explicit property access while still storing the
    % property within the cfg structure. Since the user-facing prop is
    % SetObservable, UI changes can be triggered immediately.
    %
    % Finally, a small subset of the most important configuration params 
    % (eg labelMode, nview) have their own nonDependent properties.
    % When configurations are loaded/saved, these props are explicitly 
    % copyied from/to the configuration struct.
    %
    % Maybe an improved/ultimate implementation:
    % * Make .projPrefs and all subprops handle objects. This improves the
    % set API by i) eliminating mistyped fieldnames, ii) making set-checks
    % convenient, and iii) making set-observation and UI updates
    % convenient.
    % * Eliminate the 2nd set of params above, just use the .projPrefs
    % structure which now has listeners attached.
    % * The most important configuration params can still have their own
    % props.
    projPrefs; % init: C
    
    projVerbose = 0; % transient, unmanaged
    
    isgui = true; % whether there is a GUI
    unTarLoc = ''; % location that project has most recently been untarred to
    
    projRngSeed = 17;
  end
  properties (Dependent)
    hasProject            % scalar logical
    projectfile;          % Full path to current project 
    projectroot;          % Parent dir of projectfile, if it exists
  end

  %% Movie/Video
  % Originally "Movie" referred to high-level data units/elements, eg
  % things added, removed, managed by the MovieManager etc; while "Video"
  % referred to visual details, display, playback, etc. But I sort of
  % forgot and mixed them up so that Movie sometimes applies to the latter.
  properties (SetAccess=private)
    nview; % number of views. init: C
    viewNames % [nview] cellstr. init: C
    
    % States of viewCalProjWide/viewCalData:
    % .viewCalProjWide=[], .vCD=any. Here .vcPW is uninitted and .vCD is unset/immaterial.
    % .viewCalProjWide=true, .vCD=<scalar Calrig obj>. Scalar Calrig obj apples to all movies, including GT
    % .viewCalProjWide=false, .vCD=[nMovSet] cell array of calRigs. .vCD
    % applies element-wise to movies. .vCD{i} can be empty indicating unset
    % calibration object for that movie.
    viewCalProjWide % [], true, or false. init: PN
    viewCalibrationData % Opaque calibration 'useradata' for multiview. init: PN
    viewCalibrationDataGT % etc. 
    
    movieReadPreLoadMovies = false; % scalar logical. Set .preload property on any MovieReaders per this prop
    movieReader = []; % [1xnview] MovieReader objects. init: C
    movieInfoAll = {}; % cell-of-structs, same size as movieFilesAll
    movieInfoAllGT = {}; % same as .movieInfoAll but for GT mode
    movieDontAskRmMovieWithLabels = false; % If true, won't warn about removing-movies-with-labels    
    projectHasTrx = false; % whether there are trx files for any movie
  end
  properties (Dependent)
    movieInfoAllGTaware; 
    viewCalibrationDataGTaware % Either viewCalData or viewCalDataGT
    viewCalibrationDataCurrent % view calibration data applicable to current movie (gt aware)
  end
  properties (SetObservable)
    movieFilesAll = {}; % [nmovset x nview] column cellstr, full paths to movies; can include macros 
    movieFilesAllGT = {}; % same as .movieFilesAll but for GT mode
  end
  properties
    % Using cells here so movies do not have to all have the same bitDepth
    % See HistEq.genHistEqLUT for notes on how to apply LUTs
    %
    % These should prob be called "preProcMovieFilesAllHistEqLUT" since
    % they are preproc-parameter dependent etc
    movieFilesAllHistEqLUT % [nmovset x nview] cell. Each el is a scalar struct containing lut + related info, or [] 
    movieFilesAllGTHistEqLUT % [nmovsetGT x nview] "
    cmax_auto = nan(0,1);
    clim_manual = zeros(0,2);
  end
  properties (SetObservable,AbortSet)
    movieFilesAllHaveLbls = false(0,1); % [nmovsetx1] logical. 
        % How MFAHL is maintained
        % - At project load, it is updated fully.
        % - Trivial update on movieRm/movieAdd.
        % - Otherwise, all labeling operations can only affect the current
        % movie; meanwhile the FrameTable contains all necessary info to
        % update movieFilesAllHaveLbls. So we piggyback onto
        % updateFrameTable*(). 
        %
        % For MultiView, MFAHL is true if any movie in a movieset has
        % labels.
        
    movieFilesAllGTHaveLbls = false(0,1); % etc
  end
  properties (SetObservable)
    moviename; % short 'pretty' name, cosmetic purposes only. For multiview, primary movie name.
    movieCenterOnTarget = false; % scalar logical.
    movieRotateTargetUp = false;
    movieCenterOnTargetLandmark = false; % scalar logical. If true, see movieCenterOnTargetIpt. Transient, unmanaged.
    movieCenterOnTargetIpt = []; % scalar point index, used if movieCenterOnTargetLandmark=true. Transient, unmanaged
    movieForceGrayscale = false; % scalar logical. In future could make [1xnview].
    movieFrameStepBig; % scalar positive int
    movieShiftArrowNavMode; % scalar ShiftArrowMovieNavMode
  end
  properties (SetAccess=private)
    movieShiftArrowNavModeThresh; % scalar double. This is separate prop from the ShiftArrowMode so it persists even if the ShiftArrowMode changes.
  end
  properties (SetObservable)
    movieShiftArrowNavModeThreshCmp; % char, eg '<' or '>='
    moviePlaySegRadius; % scalar int
    moviePlayFPS; 
    movieInvert; % [1xnview] logical. If true, movie should be inverted when read. This is to compensate for codec issues where movies can be read inverted on platform A wrt platform B
      % Not much care is taken wrt interactions with cropInfo. If you 
      % change your .movieInvert, then your crops will likely be wrong.
      % A warning is thrown but nothing else
    
    movieViewBGsubbed = false; % transient
    
    movieIsPlaying = false;
  end
  properties (Dependent)
    isMultiView;
    movieFilesAllGTaware;
    movieFilesAllFull; % like movieFilesAll, but macro-replaced and platformized
    movieFilesAllGTFull; % etc
    movieFilesAllFullGTaware;
    movieFilesAllHaveLblsGTaware;
    hasMovie;
    moviefile;
    nframes;
    movierawnr; % [nview]. numRows in original/raw movies
    movierawnc; % [nview]. numCols in original/raw movies
    movienr; % [nview]. always equal to numRows in .movieroi
    movienc; % [nview]. always equal to numCols in .movieroi
    movieroi; % [nview x 4]. Each row is [xlo xhi ylo yhi]. If no crop present, then this is just [1 nc 1 nr].
    movieroictr; % [nview x 2]. Each row is [xc yc] center of current roi in that view.
    nmovies;
    nmoviesGT;
    nmoviesGTaware;
    moviesSelected; % [nSel] vector of MovieIndices currently selected in MovieManager. GT mode ok.
  end
  
  %% Crop
  properties
    movieFilesAllCropInfo % [nmovset x 1] cell. Each el is a [nview] array of cropInfos, or [] if no crop info 
    movieFilesAllGTCropInfo % [nmovsetGT x 1] "
    cropIsCropMode % scalar logical
  end
  properties (Dependent)
    movieFilesAllCropInfoGTaware
    cropProjHasCrops % scalar logical. If true, all elements of movieFilesAll*CropInfo are populated. If false, all elements of " are []
  end
  events
    cropIsCropModeChanged % cropIsCropMode mutated
    cropCropsChanged % something in .movieFilesAll*CropInfo mutated
    cropUpdateCropGUITools
  end
  
  %% Trx
  properties (SetObservable)
    trxFilesAll = {};  % column cellstr, full paths to trxs. Same size as movieFilesAll.
    trxFilesAllGT = {}; % etc. Same size as movieFilesAllGT.
  end
  properties (SetAccess=private)
    trxCache = [];            % containers.Map. Keys: fullpath. vals: lazy-loaded structs with fields: .trx and .frm2trx
    trx = [];                 % trx object
    frm2trx = [];             % nFrm x nTrx logical. frm2trx(iFrm,iTrx) is true if trx iTrx is live on frame iFrm (for current movie)
    tblTrxData = [];          % last-used data in tblTrx
  end
  properties (Dependent,SetObservable)
    targetZoomRadiusDefault;
  end
  properties (Dependent)
    trxFilesAllFull % like .movieFilesAllFull, but for .trxFilesAll
    trxFilesAllGTFull % etc
    trxFilesAllFullGTaware
    hasTrx
    currTrx
    nTrx
    nTargets % nTrx, or 1 if no Trx
  end
  
  %% ShowTrx
  properties (SetObservable)
    showTrx;                  % true to show trajectories
    showTrxCurrTargetOnly;    % if true, plot only current target
    showTrxIDLbl;             % true to show id label 
    showOccludedBox;          % whether to show the occluded box
    
    showSkeleton;           % true to plot skeleton 
  end 
  properties
    hTraj;                    % nTrx x 1 vector of line handles
    hTrx;                     % nTrx x 1 vector of line handles
    hTrxTxt;                  % nTrx x 1 vector of text handles
    showTrxPreNFrm = 15;      % number of preceding frames to show in traj
    showTrxPostNFrm = 5;      % number of following frames to show in traj
  end
  
  %% Labeling
  properties (SetObservable)
    labelMode;            % scalar LabelMode. init: C
    labeledpos;           % column cell vec with .nmovies elements. labeledpos{iMov} is npts x 2 x nFrm(iMov) x nTrx(iMov) double array; labeledpos{1}(:,1,:,:) is X-coord, labeledpos{1}(:,2,:,:) is Y-coord. init: PN
    % Multiview. Right now all 3d pts must live in all views, eg
    % .nLabelPoints=nView*NumLabelPoints. first dim of labeledpos is
    % ordered as {pt1vw1,pt2vw1,...ptNvw1,pt1vw2,...ptNvwK}
    labeledposTS;         % labeledposTS{iMov} is nptsxnFrm(iMov)xnTrx(iMov). It is the last time .labeledpos or .labeledpostag was touched. init: PN
    labeledposMarked;     % labeledposMarked{iMov} is a nptsxnFrm(iMov)xnTrx(iMov) logical array. Elements are set to true when the corresponding pts have their labels set; users can set elements to false at random. init: PN
    labeledpostag;        % column cell vec with .nmovies elements. labeledpostag{iMov} is npts x nFrm(iMov) x nTrx(iMov) logical indicating *occludedness*. ("tag" for legacy reasons) init: PN
    
    labeledpos2;          % identical size/shape with labeledpos. aux labels (eg predicted, 2nd set, etc). init: PN
    labels2Hide;          % scalar logical
    labels2ShowCurrTargetOnly;  % scalar logical, transient
    
    labeledposGT          % like .labeledpos    
    labeledposTSGT        % like .labeledposTS
    labeledpostagGT       % like .labeledpostag
    labeledpos2GT         % like .labeledpos2

    skeletonEdges = zeros(0,2); % nEdges x 2 matrix containing indices of vertex landmarks
    flipLandmarkMatches = zeros(0,2); % nPairs x 2 matrix containing indices of vertex landmarks
    
  end
  properties % make public setaccess
    labelPointsPlotInfo;  % struct containing cosmetic info for labelPoints. init: C
    predPointsPlotInfo;  % " predicted points. init: C
    impPointsPlotInfo;
  end
  properties (SetAccess=private)
    nLabelPoints;         % scalar integer. This is the total number of 2D labeled points across all views. Contrast with nPhysPoints. init: C
    labelTemplate;
    
    labeledposIPtSetMap;  % [nptsets x nview] 3d 'point set' identifications. labeledposIPtSetMap(iSet,:) gives
                          % point indices for set iSet in various views. init: C
    labeledposSetNames;   % [nptsets] cellstr names labeling rows of .labeledposIPtSetMap.
                          % NOTE: arguably the "point names" should be. init: C
    labeledposIPt2View;   % [npts] vector of indices into 1:obj.nview. Convenience prop, derived from .labeledposIPtSetMap. init: C
    labeledposIPt2Set;    % [npts] vector of set indices for each point. Convenience prop. init: C
  end
  properties (SetObservable)
    labeledposNeedsSave;  % scalar logical, .labeledpos has been touched since last save. Currently does NOT account for labeledpostag
    lastLabelChangeTS     % last time training labels were changed
    needsSave; 
  end
  properties (Dependent)
    labeledposGTaware;
    labeledposTSGTaware;
    labeledpostagGTaware;
    labeledpos2GTaware;
    labeledposCurrMovie;
    labeledpos2CurrMovie;
    labeledpostagCurrMovie;
    
    nPhysPoints; % number of physical/3D points
  end
  properties (SetObservable)
    lblCore; % init: L
  end
  properties
    labeledpos2trkViz % scalar TrackingVisualizerMT
  end
  
  properties
    fgEmpiricalPDF % struct containing empirical FG pdf and metadata
  end
  
  %% GT mode
  properties (SetObservable,SetAccess=private)
    gtIsGTMode % scalar logical
  end
  properties
    gtSuggMFTable % [nGTSugg x ncol] MFTable for suggested frames to label. .mov values are MovieIndexes
    gtSuggMFTableLbled % [nGTSuggx1] logical flags indicating whether rows of .gtSuggMFTable were gt-labeled

    gtTblRes % [nGTcomp x ncol] table, or []. Most recent GT performance results. 
      % gtTblRes(:,MFTable.FLDSID) need not match
      % gtSuggMFTable(:,MFTable.FLDSID) because eg GT performance can be 
      % computed even if some suggested frames are not be labeled.
  end
  properties (Dependent)
    gtNumSugg % height(gtSuggMFTable)
  end
  events
    % Instead of making gtSuggMFTable* SetObservable, we use these events.
    % The two variables are coupled (number of rows must be equal,
    % so updating them is a nonatomic (2-step) process. Listeners directly
    % listening to property sets will sometimes see inconsistent state.
    
    gtIsGTModeChanged 
    gtSuggUpdated % general update occurred of gtSuggMFTable*
    gtSuggMFTableLbledUpdated % incremental update of gtSuggMFTableLbled occurred
    gtResUpdated % update of GT performance results occurred
  end
  
  
  %% Suspiciousness
  properties (SetObservable,SetAccess=private)
    suspScore; % column cell vec same size as labeledpos. suspScore{iMov} is nFrm(iMov) x nTrx(iMov)
    suspSelectedMFT; % MFT table of selected suspicous frames.
    suspComputeFcn; 
    % Function with sig [score,tblMFT,diagstr]=fcn(labelerObj) that 
    % computes suspScore, suspSelectedMFT.
    % See .suspScore for required size/dims of suspScore and contents.
    % diagstr is arbitrary diagnostic info (assumed char for now).
    
    suspDiag; % Transient "userdata", diagnostic output from suspComputeFcn
    
%     currSusp; % suspScore for current mov/frm/tgt. Can be [] indicating 'N/A'
    %     suspNotes; % column cell vec same size as labeledpos. suspNotes{iMov} is a nFrm x nTrx column cellstr
  end
  
  %% PreProc
  properties
    preProcH0 % Either [], or a struct with field .hgram which is [nbin x nview]. Conceptually, this is a preProcParam that APT updates from movies
    preProcData % scalar CPRData, preproc Data cache for CPR
    preProcDataTS % scalar timestamp  
    preProcSaveData % scalar logical. If true, preProcData* and ppdb are saved/loaded with project file
    copyPreProcData = false; % scalar logical. if true, don't reread images from videos to create the PreProcDB, just copy over from preProcData
    
    ppdb % PreProcDB for DL
  end

  properties (Dependent)
    
    preProcParams % struct - KB 20190214 -- made this a dependent property, derived from trackParams

  end  
  %% Tracking
  properties (SetObservable)
    trackersAll % cell vec of concrete LabelTracker objects. init: PNPL
    currTracker % scalar int, either 0 for "no tracker" or index into trackersAll
  end
  properties (Dependent)
    tracker % The current tracker, or []
    trackerAlgo % The current tracker algorithm, or ''
    trackerIsDL
    trackDLParams % scalar struct, common DL params
%     cprParams % scalar struct, cpr parameters
    DLCacheDir % string, location of DL cache dir
  end
  properties (SetObservable)
    trackModeIdx % index into MFTSetEnum.TrackingMenu* for current trackmode. 
     %Note MFTSetEnum.TrackingMenuNoTrx==MFTSetEnum.TrackingMenuTrx(1:K).
     %Values of trackModeIdx 1..K apply to either the NoTrx or Trx cases; 
     %larger values apply only the Trx case.
     
    trackDLBackEnd % scalar DLBackEndClass
    
    trackNFramesSmall % small/fine frame increment for tracking. init: C
    trackNFramesLarge % big/coarse ". init: C
    trackNFramesNear % neighborhood radius. init: C
    trackParams; % all tracking parameters. init: C
  end
  properties
    trkResIDs % [nTR x 1] cellstr unique IDs
    trkRes % [nMov x nview x nTR] cell. cell array of TrkFile objs
    trkResGT % [nMovGT x nview x nTR] cell. etc
    trkResViz % [nTR x 1] cell. TrackingVisualizer vector
  end
  properties (Dependent)
    trkResGTaware
  end
  
  %% CrossValidation
  properties
    xvResults % table of most recent cross-validation results. This table
      % has a row for every labeled frame present at the time xvalidation 
      % was run. So it should be fairly explicit if/when it is out-of-date 
      % relative to the project (eg when labels are added or removed)
    xvResultsTS % timestamp for xvResults
  end
  
  %% Prev
  properties
    prevIm = struct('CData',0,'XData',0,'YData',0); % struct, like a stripped image handle (.CData, .XData, .YData). 'primary' view only
    prevAxesMode; % scalar PrevAxesMode
    prevAxesModeInfo; % "userdata" for .prevAxesMode
    lblPrev_ptsH; % [npts] gobjects. init: L
    lblPrev_ptsTxtH; % [npts] etc. init: L
  end
  
  %% Misc
  properties (SetObservable, AbortSet)
    prevFrame = nan;      % last previously VISITED frame
    currTarget = 1;     % always 1 if proj doesn't have trx
    
    currImHud; % scalar AxisHUD object TODO: move to LabelerGUI. init: C
  end
  properties (SetObservable)
    keyPressHandlers; % [nhandlerx1] cell array of LabelerKeyEventHandlers.
  end
  properties (AbortSet)
    currMovie; % idx into .movieFilesAll (row index, when obj.multiView is true), or .movieFilesAllGT when .gtIsGTmode is on
    % Don't observe this, listen to 'newMovie'
  end
  properties (Dependent)
    currMovIdx; % scalar MovieIndex
  end
  properties 
    currFrame = 1; % current frame
    currIm = [];            % [nview] cell vec of image data. init: C
    selectedFrames = [];    % vector of frames currently selected frames; typically t0:t1
    hFig; % handle to main LabelerGUI figure
  end
  properties (SetAccess=private)
    isinit = false;         % scalar logical; true during initialization, when some invariants not respected
  end
  properties (Dependent)
    gdata; % handles structure for LabelerGUI
  end

  
  %% Prop access
  methods % dependent prop getters
    function v = get.viewCalibrationDataGTaware(obj)
      v = obj.getViewCalibrationDataGTawareArg(obj.gtIsGTMode);
    end
    function v = getViewCalibrationDataGTawareArg(obj,gt)
      if gt
        v = obj.viewCalibrationDataGT;
      else
        v = obj.viewCalibrationData;
      end
    end
    function v = get.viewCalibrationDataCurrent(obj)
      % Nearly a forward to getViewCalibrationDataMovIdx except edge cases
      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        v = [];
      elseif vcdPW
        vcd = obj.viewCalibrationData; % applies to regular and GT movs
        assert(isequal(vcd,[]) || isscalar(vcd));
        v = vcd;
      else % ~vcdPW
        vcd = obj.viewCalibrationDataGTaware;
        nmov = obj.nmoviesGTaware;
        assert(iscell(vcd) && numel(vcd)==nmov);
        if nmov==0 || obj.currMovie==0
          v = [];
        else
          v = vcd{obj.currMovie};
        end
      end
    end
    function v = getViewCalibrationDataMovIdx(obj,mIdx)
      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        v = [];
      elseif vcdPW
        vcd = obj.viewCalibrationData; % applies to regular and GT movs
        assert(isequal(vcd,[]) || isscalar(vcd));
        v = vcd;
      else % ~vcdPW
        [iMov,gt] = mIdx.get();
        vcd = obj.getViewCalibrationDataGTawareArg(gt);
        nmov = obj.getnmoviesGTawareArg(gt);
        assert(iscell(vcd) && numel(vcd)==nmov);
        v = vcd{iMov};
      end
    end
    function v = get.isMultiView(obj)
      v = obj.nview>1;
    end
    function v = get.movieFilesAllGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieFilesAllGT;
      else        
        v = obj.movieFilesAll;
      end
    end
    function v = get.movieFilesAllFull(obj)
      % See also .projLocalizePath()
      sMacro = obj.projMacrosGetWithAuto();
      v = FSPath.fullyLocalizeStandardize(obj.movieFilesAll,sMacro);
      FSPath.warnUnreplacedMacros(v);
    end
    function v = get.movieFilesAllGTFull(obj)
      sMacro = obj.projMacrosGetWithAuto();
      v = FSPath.fullyLocalizeStandardize(obj.movieFilesAllGT,sMacro);
      FSPath.warnUnreplacedMacros(v);
    end
    function v = get.movieFilesAllFullGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieFilesAllGTFull;
      else        
        v = obj.movieFilesAllFull;
      end
    end
    function v = getMovieFilesAllFullMovIdx(obj,mIdx)
      % mIdx: MovieIndex vector
      % v: [numel(mIdx)xnview] movieFilesAllFull/GT 
      
      assert(isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      n = numel(iMov);
      v = cell(n,obj.nview);
      mfaf = obj.movieFilesAllFull;
      mfafGT = obj.movieFilesAllGTFull;
      for i=1:n
        if gt(i)
          v(i,:) = mfafGT(iMov(i),:);
        else
          v(i,:) = mfaf(iMov(i),:);
        end
      end
    end
    function [tffound,mIdx] = getMovIdxMovieFilesAllFull(obj,movsets,varargin)
      % Get the MovieIndex vector corresponding to a set of movies by
      % comparing against .movieFilesAll*Full.
      %
      % Defaults to using .movieFilesAll*Full per .gtIsGTMode.
      %
      % movsets: [n x nview] movie fullpaths. Can have repeated
      % rows/moviesets.
      %
      % tffound: [n x 1] logical
      % mIdx: [n x 1] MovieIndex vector. mIdx(i)==0<=>tffound(i)==false.
      
      gt = myparse(varargin,...
        'gt',obj.gtIsGTMode);
      
      if gt
        mfaf = obj.movieFilesAllGTFull;
      else        
        mfaf = obj.movieFilesAllFull;
      end
      
      [nmovset,nvw] = size(movsets);
      assert(nvw==obj.nview);
      
      movsets = FSPath.standardPath(movsets);
      movsets = cellfun(@FSPath.platformizePath,movsets,'uni',0);
      [iMov1,iMov2] = Labeler.identifyCommonMovSets(movsets,mfaf);
      
      tffound = false(nmovset,1);
      tffound(iMov1,:) = true;
      mIdx = zeros(nmovset,1);
      mIdx(iMov1) = iMov2; % mIdx is all 0s, or positive movie indices
      mIdx = MovieIndex(mIdx,gt);
    end
    function v = get.movieFilesAllHaveLblsGTaware(obj)
      v = obj.getMovieFilesAllHaveLblsArg(obj.gtIsGTMode);
    end
    function v = getMovieFilesAllHaveLblsArg(obj,gt)
      if gt
        v = obj.movieFilesAllGTHaveLbls;
      else
        v = obj.movieFilesAllHaveLbls;
      end      
    end
    function v = get.movieFilesAllCropInfoGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieFilesAllGTCropInfo;
      else
        v = obj.movieFilesAllCropInfo;
      end
    end
    function v = getMovieFilesAllHistEqLUTGTawareStc(obj,gt)
      if gt
        v = obj.movieFilesAllGTHistEqLUT;
      else
        v = obj.movieFilesAllHistEqLUT;
      end
    end
    function v = getMovieFilesAllHistEqLUTMovIdx(obj,mIdx)
      % v: [1xnview] row of .movieFilesAll*HistEqLUT
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [imov,gt] = mIdx.get();
      if gt
        v = obj.movieFilesAllGTHistEqLUT(imov,:);
      else
        v = obj.movieFilesAllHistEqLUT(imov,:);
      end
    end
    function v = get.cropProjHasCrops(obj)
      % Returns true if proj has at least one movie and has crops
      v = obj.nmovies>0 && ~isempty(obj.movieFilesAllCropInfo{1});
      
%       ci = [obj.movieFilesAllCropInfo; obj.movieFilesAllGTCropInfo];
%       tf = ~cellfun(@isempty,ci);
%       v = all(tf);
%       assert(v || ~any(tf));
    end
    function v = getMovieFilesAllCropInfoMovIdx(obj,mIdx)
      % mIdx: scalar MovieIndex 
      % v: empty, or [nview] CropInfo array 
      
      assert(isa(mIdx,'MovieIndex'));
      assert(isscalar(mIdx));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.movieFilesAllGTCropInfo{iMov};
      else
        v = obj.movieFilesAllCropInfo{iMov};
      end
    end
    function v = get.trxFilesAllFull(obj)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.
      v = Labeler.trxFilesLocalize(obj.trxFilesAll,obj.movieFilesAllFull);
    end    
    function v = get.trxFilesAllGTFull(obj)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.
      v = Labeler.trxFilesLocalize(obj.trxFilesAllGT,obj.movieFilesAllGTFull);
    end
    function v = get.trxFilesAllFullGTaware(obj)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.
      if obj.gtIsGTMode
        v = obj.trxFilesAllGTFull;
      else
        v = obj.trxFilesAllFull;
      end
    end
    function v = getTrxFilesAllFullMovIdx(obj,mIdx)
      % Warning: Expensive to call. Call me once and then index rather than
      % using a compound indexing-expr.      
      assert(all(isa(mIdx,'MovieIndex')));
      [iMov,gt] = mIdx.get();
      n = numel(iMov);
      v = cell(n,obj.nview);
      tfaf = obj.trxFilesAllFull;
      tfafGT = obj.trxFilesAllGTFull;
      for i=1:n
        if gt
          v(i,:) = tfafGT(iMov(i),:);
        else
          v(i,:) = tfaf(iMov(i),:);
        end
      end
    end
    
    function v = get.hasMovie(obj)
      v = obj.hasProject && obj.movieReader(1).isOpen;
    end    
    function v = get.moviefile(obj)
      mr = obj.movieReader(1);
      if isempty(mr)
        v = [];
      else
        v = mr.filename;
      end
    end
    function v = get.movierawnr(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = [mr.nr]';
      else
        v = nan(obj.nview,1);
      end
    end
    function v = get.movierawnc(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = [mr.nc]';          
      else
        v = nan(obj.nview,1);
      end
    end    
    function v = get.movienr(obj)
      v = obj.movieroi;
      v = v(:,4)-v(:,3)+1;
    end
    function v = get.movienc(obj)
      v = obj.movieroi;
      v = v(:,2)-v(:,1)+1;
    end    
    function v = get.movieroi(obj)
      mr = obj.movieReader;
      if mr(1).isOpen
        v = cat(1,mr.roiread);
      else
        v = nan(obj.nview,4);
      end
    end
    function v = get.movieroictr(obj)
      rois = obj.movieroi;
      v = [rois(:,1)+rois(:,2),rois(:,3)+rois(:,4)]/2;
    end
    function rois = getMovieRoiMovIdx(obj,mIdx)
      % v: [nview x 4] roi
      
      if obj.cropProjHasCrops
        ci = obj.getMovieFilesAllCropInfoMovIdx(mIdx);
        assert(~isempty(ci));        
        rois = cat(1,ci.roi);
      else
        [iMov,gt] = mIdx.get();
        mia = obj.getMovieInfoAllGTawareArg(gt);
        mia = mia(iMov,:);
        rois = cellfun(@(x)[1 x.info.nc 1 x.info.nr],mia,'uni',0);
        rois = cat(1,rois{:});
      end
      szassert(rois,[obj.nview 4]);
    end
    function v = get.nframes(obj)
      if isempty(obj.currMovie) || obj.currMovie==0
        v = nan;
      else
        % multiview case: ifos have .nframes set identically if movies have
        % different lengths
        ifo = obj.movieInfoAllGTaware{obj.currMovie,1};
        v = ifo.nframes;
      end
    end
    function [ncmin,nrmin] = getMinMovieWidthHeight(obj)
      movInfos = [obj.movieInfoAll; obj.movieInfoAllGT];
      nrall = cellfun(@(x)x.info.nr,movInfos); % [(nmov+nmovGT) x nview]
      ncall = cellfun(@(x)x.info.nc,movInfos); % etc
      nrmin = min(nrall,[],1); % [1xnview]
      ncmin = min(ncall,[],1); % [1xnview]      
    end
    function v = getNFramesMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt        
        movInfo = obj.movieInfoAllGT{iMov,1};
      else
        movInfo = obj.movieInfoAll{iMov,1};
      end
      v = movInfo.nframes;
    end
    function v = getNFramesMovFile(obj,movFile)
      
      movfilefull = obj.projLocalizePath(movFile);
      assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
      
      mr = MovieReader;
      mr.open(movfilefull);
      v = mr.nframes;
      mr.close();
    end
    
    
    function v = get.moviesSelected(obj) %#%GUIREQ
      % Find MovieManager in LabelerGUI
      
      handles = obj.gdata;
      if isfield(handles,'movieMgr')
        mmc = handles.movieMgr;
      else
        mmc = [];
      end
      if ~isempty(mmc) && isvalid(mmc)
        v = mmc.getSelectedMovies();
      else
        error('Labeler:getMoviesSelected',...
          'Cannot access Movie Manager. Make sure your desired movies are selected in the Movie Manager.');
      end
    end
    function v = get.hasTrx(obj)
      v = ~isempty(obj.trx);
    end
    function v = get.currTrx(obj)
      if obj.hasTrx
        v = obj.trx(obj.currTarget);
      else
        v = [];
      end
    end
    function v = get.nTrx(obj)
      v = numel(obj.trx);
    end
    function v = getnTrxMovIdx(obj,mIdx)
      % Warning: Slow/expensive, call in bulk and index rather than calling
      % piecemeal 
      %
      % mIdx: [n] MovieIndex vector
      % v: [n] array; v(i) == num targets in ith mov
      
      assert(isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      assert(all(gt) || ~any(gt));
      gt = all(gt);
      
      PROPS = obj.gtGetSharedPropsStc(gt);
      tfaf = obj.(PROPS.TFAF)(iMov,1);
      mia = obj.(PROPS.MIA)(iMov,1);
      v = ones(size(iMov));
      for i=1:numel(v)
        trxfile = tfaf{i};
        if isempty(trxfile)
          % none; v(i) is 1
        else
          nfrm = mia{i}.nframes;
          trxI = obj.getTrx(trxfile,nfrm);
          v(i) = numel(trxI);
        end
      end
    end
    function v = get.nTargets(obj)
      if obj.hasTrx
        v = obj.nTrx;
      else
        v = 1;
      end
    end
    function v = get.targetZoomRadiusDefault(obj)
      v = obj.projPrefs.Trx.ZoomFactorDefault;
    end
    function v = get.hasProject(obj)
      % AL 20160710: debateable utility/correctness, but if you try to do
      % some things (eg open MovieManager) right on bootup from an empty
      % Labeler you get weird errors.
      v = size(obj.movieFilesAll,2)>0; % AL 201806: want first dim instead?
    end
    function v = get.projectfile(obj)
      info = obj.projFSInfo;
      if ~isempty(info)
        v = info.filename;
      else
        v = [];
      end
    end
    function v = get.projectroot(obj)
      v = obj.projectfile;
      if ~isempty(v)
        v = fileparts(v);
      end
    end
    function v = get.nmovies(obj)
      % for multiview labeling, this is really 'nmoviesets'
      v = size(obj.movieFilesAll,1);
    end
    function v = get.nmoviesGT(obj)
      v = size(obj.movieFilesAllGT,1);
    end
    function v = get.nmoviesGTaware(obj)
      v = obj.getnmoviesGTawareArg(obj.gtIsGTMode);
    end
    function v = getnmoviesGTawareArg(obj,gt)
      if ~gt
        v = size(obj.movieFilesAll,1);
      else
        v = size(obj.movieFilesAllGT,1);
      end
    end
    function v = get.movieInfoAllGTaware(obj)
      if obj.gtIsGTMode
        v = obj.movieInfoAllGT;
      else
        v = obj.movieInfoAll;
      end
    end
    function v = getMovieInfoAllGTawareArg(obj,gt)
      if gt
        v = obj.movieInfoAllGT;
      else
        v = obj.movieInfoAll;
      end
    end
    function v = get.labeledposGTaware(obj)
      v = obj.getlabeledposGTawareArg(obj.gtIsGTMode);
    end
    function v = getlabeledposGTawareArg(obj,gt)
      if gt
        v = obj.labeledposGT;
      else
        v = obj.labeledpos;
      end
    end
    function v = getLabeledPosMovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labeledposGT{iMov};
      else 
        v = obj.labeledpos{iMov};
      end
    end
    function v = getLabeledPos2MovIdx(obj,mIdx)
      assert(isscalar(mIdx) && isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt
        v = obj.labeledpos2GT{iMov};
      else 
        v = obj.labeledpos2{iMov};
      end
    end
    function v = get.labeledposTSGTaware(obj)
      if obj.gtIsGTMode
        v = obj.labeledposTSGT;
      else
        v = obj.labeledposTS;
      end
    end
    function v = get.labeledpostagGTaware(obj)
      if obj.gtIsGTMode
        v = obj.labeledpostagGT;
      else
        v = obj.labeledpostag;
      end
    end
    function v = get.labeledpos2GTaware(obj)
      if obj.gtIsGTMode
        v = obj.labeledpos2GT;
      else
        v = obj.labeledpos2;
      end
    end
    function v = get.labeledposCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledposGT{obj.currMovie};
      else
        v = obj.labeledpos{obj.currMovie};
      end
    end
    function v = get.labeledpos2CurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledpos2GT{obj.currMovie};
      else
        v = obj.labeledpos2{obj.currMovie};
      end
    end
    function v = get.labeledpostagCurrMovie(obj)
      if obj.currMovie==0
        v = [];
      elseif obj.gtIsGTMode
        v = obj.labeledpostagGT{obj.currMovie};
      else
        v = obj.labeledpostag{obj.currMovie};
      end
    end
    function v = get.nPhysPoints(obj)
      v = size(obj.labeledposIPtSetMap,1);
    end
    function v = get.currMovIdx(obj)
      v = MovieIndex(obj.currMovie,obj.gtIsGTMode);
    end
    function v = get.gdata(obj)
      v = guidata(obj.hFig);
    end
    function v = get.gtNumSugg(obj)
      v = height(obj.gtSuggMFTable);
    end
    function v = get.tracker(obj)
      if obj.currTracker==0 || isempty(obj.trackersAll),
        v = [];
      else
        v = obj.trackersAll{obj.currTracker};
      end
    end
    function v = get.trackerAlgo(obj)
      v = obj.tracker;
      if isempty(v)
        v = '';
      else
        v = v.algorithmName;
      end
    end
    function v = get.trackerIsDL(obj)
      v = obj.tracker;
      if isempty(v)
        v = [];
      else
        v = isa(v,'DeepTracker');
      end
    end
    % KB 20190214 - store all parameters together in one struct. these dependent functions emulate previous behavior
    function v = get.preProcParams(obj)
      
      if isempty(obj.trackParams),
        v = [];
      else
        v = APTParameters.all2PreProcParams(obj.trackParams);
      end

    end
    
    function v = get.trackDLParams(obj)
      
      if isempty(obj.trackParams),
        v = [];
      else
        v = APTParameters.all2TrackDLParams(obj.trackParams);
      end

    end
    
    function v = get.DLCacheDir(obj)      
      % AL 20190425 bundle. .DLCacheDir is immutable during a Session and 
      % always equal to .projTempDir
      v = obj.projTempDir;
%       if isempty(obj.trackParams),
%         v = '';
%       else
%         v = APTParameters.all2DLCacheDir(obj.trackParams);
%       end
    end
    
%     function v = get.cprParams(obj)      
%       if isempty(obj.trackParams),
%         v = [];
%       else
%         v = APTParameters.all2CPRParams(obj.trackParams,obj.nPhysPoints,obj.nview);
%       end      
%     end
    
%     function set.cprParams(obj,prmCpr)
%       
%       if isempty(obj.trackParams),
%         obj.trackParams = struct;
%         obj.trackParams.ROOT = APTParameters.defaultParamsStruct;
%       end
%       obj.trackParams.ROOT.CPR = CPRParam.old2newCPROnly(prmCpr);
%       
%     end
    function v = get.trkResGTaware(obj)
      gt = obj.gtIsGTMode;
      if gt
        v = obj.trkResGT;
      else
        v = obj.trkRes;
      end
    end
  end
  
  methods % prop access
    % CONSIDER get rid of setter, use listeners
    function set.labeledpos(obj,v)
      obj.labeledpos = v;
      if ~obj.isinit %#ok<MCSUP> 
        obj.updateTrxTable();
        obj.updateFrameTableIncremental(); 
      end
    end
    function set.labeledposGT(obj,v)
      obj.labeledposGT = v;
      if ~obj.isinit %#ok<MCSUP> 
        obj.updateTrxTable();
        obj.updateFrameTableIncremental();
        obj.gtUpdateSuggMFTableLbledIncremental();
      end
    end
    function set.movieForceGrayscale(obj,v)
      assert(isscalar(v) && islogical(v));
      [obj.movieReader.forceGrayscale] = deal(v); %#ok<MCSUP>
      obj.movieForceGrayscale = v;
    end
    function set.movieInvert(obj,v)
      assert(islogical(v) && numel(v)==obj.nview); %#ok<MCSUP>
      movieInvert0 = obj.movieInvert;
      if ~isequal(movieInvert0,v)
        mrs = obj.movieReader; %#ok<MCSUP>
        for i=1:obj.nview %#ok<MCSUP>
          mrs(i).flipVert = v(i);
        end
        if ~obj.isinit %#ok<MCSUP>
          obj.preProcNonstandardParamChanged();
          if obj.cropProjHasCrops %#ok<MCSUP>
            warningNoTrace('Project cropping will NOT be inverted/updated. Please check your crops.')
          end
        end
        obj.movieInvert = v;
      end
    end
    function set.movieViewBGsubbed(obj,v)
      assert(isscalar(v) && islogical(v));
      if v
        ppPrms = obj.preProcParams; %#ok<MCSUP>
        if isempty(ppPrms) || ...
           isempty(ppPrms.BackSub.BGType) || isempty(ppPrms.BackSub.BGReadFcn)
            error('Background type and/or background read function are not set in tracking parameters.');
        end
      end
      obj.movieViewBGsubbed = v;
      obj.hlpSetCurrPrevFrame(obj.currFrame,true); %#ok<MCSUP>
      caxis(obj.gdata.axes_curr,'auto'); %#ok<MCSUP>
    end
    function set.movieCenterOnTarget(obj,v)
      obj.movieCenterOnTarget = v;
      if ~v && obj.movieRotateTargetUp %#ok<MCSUP>
        obj.movieRotateTargetUp = false; %#ok<MCSUP>
      end
      if v
        if obj.hasTrx %#ok<MCSUP>
          obj.videoCenterOnCurrTarget();
        elseif ~obj.isinit %#ok<MCSUP>
          warningNoTrace('Labeler:trx',...
            'The current movie does not have an associated trx file. Property ''movieCenterOnTarget'' will have no effect.');
        end
      end
    end
    function set.movieRotateTargetUp(obj,v)
      if v && ~obj.movieCenterOnTarget %#ok<MCSUP>
        %warningNoTrace('Labeler:prop','Setting .movieCenterOnTarget to true.');
        obj.movieCenterOnTarget = true; %#ok<MCSUP>
      end
      obj.movieRotateTargetUp = v;
      if obj.hasTrx && obj.movieCenterOnTarget %#ok<MCSUP>
        obj.videoCenterOnCurrTarget();
      end
      if v
        if ~obj.hasTrx && ~obj.isinit %#ok<MCSUP>
          warningNoTrace('Labeler:trx',...
            'The current movie does not have an associated trx file. Property ''movieRotateTargetUp'' will have no effect.');
        end
      end
    end
    function set.targetZoomRadiusDefault(obj,v)
      obj.projPrefs.Trx.ZoomFactorDefault = v;
    end
    
    function tfIsReady = isReady(obj)
      tfIsReady = ~obj.isinit && obj.hasMovie && obj.hasProject;
    end
    
    function setMovieShiftArrowNavModeThresh(obj,v)
      assert(isscalar(v) && isnumeric(v));
      obj.movieShiftArrowNavModeThresh = v;
      tl = obj.gdata.labelTLInfo;
      tl.setStatThresh(v);
    end
  end
  
  %% Ctor/Dtor
  methods 
  
    function obj = Labeler(varargin)
      % lObj = Labeler();
      
      %APT.setpathsmart;

      [obj.isgui] = myparse_nocheck(varargin,'isgui',true);
      
      obj.NEIGHBORING_FRAME_OFFSETS = ...
                  neighborIndices(Labeler.NEIGHBORING_FRAME_MAXRADIUS);
      obj.hFig = LabelerGUI(obj);
    end
     
    function delete(obj)
% re: 180730        
%       if isvalid(obj.hFig)  % isvalid will fail if obj.hFig is empty
%         close(obj.hFig);
%         obj.hFig = [];
%       end     
      if ~isempty(obj.hFig) && isvalid(obj.hFig)
        gd = obj.gdata;
        deleteValidHandles(gd.depHandles);      
        deleteValidHandles(obj.hFig);
        obj.hFig = [];
      end
      be = obj.trackDLBackEnd;
      if ~isempty(be)
        be.shutdown();
      end
    end
    
  end
  
        
  %% Configurations
  methods (Hidden)

  % Property init legend
  % IFC: property initted during initFromConfig()
  % PNPL: property initted during projectNew() or projLoad()
  % L: property initted during labelingInit()
  % (todo) TI: property initted during trackingInit()
  %
  % There are only two ways to start working on a project.
  % 1. New/blank project: initFromConfig(), then projNew().
  % 2. Existing project: projLoad(), which is (initFromConfig(), then
  % property-initialization-equivalent-to-projNew().)
  
    function initFromConfig(obj,cfg)
      % Note: Config must be modernized
    
      isinit0 = obj.isinit;
      obj.isinit = true;
            
      % Views
      obj.nview = cfg.NumViews;
      if isempty(cfg.ViewNames)
        obj.viewNames = arrayfun(@(x)sprintf('view%d',x),1:obj.nview,'uni',0);
      else
        if numel(cfg.ViewNames)~=obj.nview
          error('Labeler:prefs',...
            'ViewNames: must specify %d names (one for each view)',obj.nview);
        end
        obj.viewNames = cfg.ViewNames;
      end
     
      npts3d = cfg.NumLabelPoints;
      npts = obj.nview*npts3d;      
      obj.nLabelPoints = npts;
      if isempty(cfg.LabelPointNames)
        cfg.LabelPointNames = arrayfun(@(x)sprintf('pt%d',x),(1:cfg.NumLabelPoints)','uni',0);
      end
      assert(numel(cfg.LabelPointNames)==cfg.NumLabelPoints);
         
      % pts, sets, views
      setnames = cfg.LabelPointNames;
      nSet = size(setnames,1);
      ipt2view = nan(npts,1);
      ipt2set = nan(npts,1);
      setmap = nan(nSet,obj.nview);
      for iSet = 1:nSet
        set = setnames{iSet};
        iPts = iSet:npts3d:npts;
        if numel(iPts)~=obj.nview
          error('Labeler:prefs',...
            'Number of point indices specified for ''%s'' does not equal number of views (%d).',set,obj.nview);
        end
        setmap(iSet,:) = iPts;
        
        iViewNZ = find(iPts>0);
        ipt2view(iPts(iViewNZ)) = iViewNZ;
        ipt2set(iPts(iViewNZ)) = iSet;
      end
      iptNotInAnyView = find(isnan(ipt2view));
      if ~isempty(iptNotInAnyView)
        iptNotInAnyView = arrayfun(@num2str,iptNotInAnyView,'uni',0);
        error('Labeler:prefs',...
          'The following points are not located in any view or set: %s',...
           String.cellstr2CommaSepList(iptNotInAnyView));
      end
      obj.labeledposIPt2View = ipt2view;
      obj.labeledposIPt2Set = ipt2set;
      obj.labeledposIPtSetMap = setmap;
      obj.labeledposSetNames = setnames;
      
      didsetlabelmode = false;
      if isfield(cfg,'LabelMode'),
        iscompatible = (cfg.NumViews==1 && cfg.LabelMode ~= LabelMode.MULTIVIEWCALIBRATED2) || ...
          (cfg.NumViews==2 && cfg.LabelMode == LabelMode.MULTIVIEWCALIBRATED2);
        if iscompatible,
          obj.labelMode = cfg.LabelMode;
          didsetlabelmode = true;
        end
      end
      if ~didsetlabelmode,
        if cfg.NumViews==1
          obj.labelMode = LabelMode.TEMPLATE;
        else
          obj.labelMode = LabelMode.MULTIVIEWCALIBRATED2;
        end
      end
      
      lpp = cfg.LabelPointsPlot;
      % Some convenience mods to .LabelPointsPlot
      % KB 20181022: Colors will now be nSet x 3
      if ~isfield(lpp,'Colors') || size(lpp.Colors,1)~=nSet
        lpp.Colors = feval(lpp.ColorMapName,nSet);
      end
      % .LabelPointsPlot invariants:
      % - lpp.ColorMapName, lpp.Colors both exist
      % - lpp.Colors is [nSet x 3]
      obj.labelPointsPlotInfo = lpp;
      % AL 20190603: updated .PredictPointsPlot reorg
      if ~isfield(cfg.Track.PredictPointsPlot,'Colors') || ...
          size(cfg.Track.PredictPointsPlot.Colors,1)~=nSet
        cfg.Track.PredictPointsPlot.Colors = ...
          feval(cfg.Track.PredictPointsPlot.ColorMapName,nSet);
      end
      if ~isfield(cfg.Track.ImportPointsPlot,'Colors') || ...
          size(cfg.Track.ImportPointsPlot.Colors,1)~=nSet
        cfg.Track.ImportPointsPlot.Colors = ...
          feval(cfg.Track.ImportPointsPlot.ColorMapName,nSet);
      end
      % .PredictPointsPlot color nvariants:
      % - ppp.ColorMapName, ppp.Colors both exist
      % - ppp.Colors is [nSet x 3]
      obj.predPointsPlotInfo = cfg.Track.PredictPointsPlot;
      obj.impPointsPlotInfo = cfg.Track.ImportPointsPlot;
            
      obj.trackNFramesSmall = cfg.Track.PredictFrameStep;
      obj.trackNFramesLarge = cfg.Track.PredictFrameStepBig;
      obj.trackNFramesNear = cfg.Track.PredictNeighborhood;
      obj.trackModeIdx = 1;
      cfg.Track = rmfield(cfg.Track,...
        {'PredictPointsPlot' 'PredictFrameStep' 'PredictFrameStepBig' ...
        'PredictNeighborhood'});
                  
      arrayfun(@delete,obj.movieReader);
      obj.movieReader = [];
      for i=obj.nview:-1:1
        mr(1,i) = MovieReader;
      end
      obj.movieReader = mr;
      obj.currIm = cell(obj.nview,1);
      delete(obj.currImHud);
      gd = obj.gdata;
      obj.currImHud = AxisHUD(gd.axes_curr.Parent,gd.axes_curr); 
      %obj.movieSetNoMovie();
      
      obj.movieForceGrayscale = logical(cfg.Movie.ForceGrayScale);
      obj.movieFrameStepBig = cfg.Movie.FrameStepBig;
      obj.movieShiftArrowNavMode = ShiftArrowMovieNavMode.(cfg.Movie.ShiftArrowNavMode);
      obj.movieShiftArrowNavModeThresh = cfg.Movie.ShiftArrowNavModeThresh;
      obj.movieShiftArrowNavModeThreshCmp = cfg.Movie.ShiftArrowNavModeThreshCmp;
      obj.moviePlaySegRadius = cfg.Movie.PlaySegmentRadius;
      obj.moviePlayFPS = cfg.Movie.PlayFPS;
           
      fldsRm = intersect(fieldnames(cfg),...
        {'NumViews' 'ViewNames' 'NumLabelPoints' 'LabelPointNames' ...
        'LabelMode' 'LabelPointsPlot' 'ProjectName' 'Movie'});
      obj.projPrefs = rmfield(cfg,fldsRm);
      % A few minor subprops of projPrefs have explicit props
      
      obj.notify('newProject');

      % order important: this needs to occur after 'newProject' event so
      % that figs are set up. (names get changed)
      movInvert = ViewConfig.getMovieInvert(cfg.View);
      obj.movieInvert = movInvert;
      obj.movieCenterOnTarget = cfg.View(1).CenterOnTarget;
      obj.movieRotateTargetUp = cfg.View(1).RotateTargetUp;
       
      obj.preProcInit();
      
      % Reset .trackersAll
      for i=1:numel(obj.trackersAll)
        % explicitly delete, conservative cleanup
        delete(obj.trackersAll{i}); % delete([]) does not err
      end
      obj.trackersAll = cell(1,0);
      % Trackers created/initted in projLoad and projNew; eg when loading,
      % the loaded .lbl knows what trackers to create.
      obj.currTracker = 0;
      
      obj.trackDLBackEnd = DLBackEndClass(DLBackEnd.Bsub);
      obj.trackParams = [];

      obj.projectHasTrx = cfg.Trx.HasTrx;
      obj.showOccludedBox = cfg.View.OccludedBox;
      
      obj.showTrx = cfg.Trx.ShowTrx;
      obj.showTrxCurrTargetOnly = cfg.Trx.ShowTrxCurrentTargetOnly;
      obj.showTrxIDLbl = cfg.Trx.ShowTrxIDLbl;
            
      obj.labels2Hide = false;
      obj.labels2ShowCurrTargetOnly = false;

      obj.skeletonEdges = zeros(0,2);
      obj.showSkeleton = false;
      obj.flipLandmarkMatches = zeros(0,2);
      
      % When starting a new proj after having an existing proj open, old 
      % state is lingering in .prevAxesModeInfo despite the next 
      % .setPrevAxesMode call due to various initialization foolishness
      obj.prevAxesModeInfo = []; 
      
      % New projs set to frozen, waiting for something to be labeled
      obj.setPrevAxesMode(PrevAxesMode.FROZEN,[]);
      
      % maybe useful to clear/reinit and shouldn't hurt
      obj.trxCache = containers.Map();
      
      if obj.isgui,
        RC.saveprop('lastProjectConfig',obj.getCurrentConfig());
      end
      
      obj.isinit = isinit0;
      
    end
    
    function cfg = getCurrentConfig(obj)
      % cfg is modernized

      cfg = obj.projPrefs;
      
      cfg.NumViews = obj.nview;
      cfg.ViewNames = obj.viewNames;
      cfg.NumLabelPoints = obj.nPhysPoints;
      cfg.LabelPointNames = obj.labeledposSetNames;
      cfg.LabelMode = char(obj.labelMode);

      % View stuff: read off current state of axes
      gd = obj.gdata;
      viewCfg = ViewConfig.readCfgOffViews(gd.figs_all,gd.axes_all,gd.axes_prev);
      for i=1:obj.nview
        viewCfg(i).InvertMovie = obj.movieInvert(i);
        viewCfg(i).CenterOnTarget = obj.movieCenterOnTarget;
        viewCfg(i).RotateTargetUp = obj.movieRotateTargetUp;
      end
      cfg.View = viewCfg;

      % misc config props that have an explicit labeler prop

      cfg.Movie = struct(...
        'ForceGrayScale',obj.movieForceGrayscale,...
        'FrameStepBig',obj.movieFrameStepBig,...
        'ShiftArrowNavMode',char(obj.movieShiftArrowNavMode),...
        'ShiftArrowNavModeThresh',obj.movieShiftArrowNavModeThresh,...
        'ShiftArrowNavModeThreshCmp',obj.movieShiftArrowNavModeThreshCmp,...
        'PlaySegmentRadius',obj.moviePlaySegRadius,...
        'PlayFPS',obj.moviePlayFPS);

      cfg.LabelPointsPlot = obj.labelPointsPlotInfo;      
      cfg.Trx.ShowTrx = obj.showTrx;
      cfg.Trx.ShowTrxCurrentTargetOnly = obj.showTrxCurrTargetOnly;
      cfg.Trx.ShowTrxIDLbl = obj.showTrxIDLbl;
      cfg.Trx.HasTrx = obj.projectHasTrx;
      
      cfg.Track.PredictFrameStep = obj.trackNFramesSmall;
      cfg.Track.PredictFrameStepBig = obj.trackNFramesLarge;
      cfg.Track.PredictNeighborhood = obj.trackNFramesNear;
      cfg.Track.PredictPointsPlot = obj.predPointsPlotInfo;
      cfg.Track.ImportPointsPlot = obj.impPointsPlotInfo;
      
      cfg.PrevAxes.Mode = char(obj.prevAxesMode);
      cfg.PrevAxes.ModeInfo = obj.prevAxesModeInfo;
    end
    
  end
    
  % Consider moving this stuff to Config.m
  methods (Static)
    
    function cfg = cfgGetLastProjectConfigNoView
      cfgBase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      cfg = RC.getprop('lastProjectConfig');
      if isempty(cfg)
        cfg = cfgBase;
      else
        cfg = structoverlay(cfgBase,cfg,'dontWarnUnrecog',true);
        
        % use "fresh"/empty View stuff on the theory that this 
        % often/usually doesn't generalize across projects
        cfg.View = repmat(cfgBase.View,cfg.NumViews,1); 
      end
    end
    
    function cfg = cfgModernizeBase(cfg)
      % 20190602 cfg.LabelPointsPlot, cfg.Track.PredictPointsPlot rearrange
      
      if ~isfield(cfg.LabelPointsPlot,'MarkerProps')
        FLDS = {'Marker' 'MarkerSize' 'LineWidth'};
        cfg.LabelPointsPlot.MarkerProps = structrestrictflds(cfg.LabelPointsPlot,FLDS);
        cfg.LabelPointsPlot = rmfield(cfg.LabelPointsPlot,FLDS);
      end
      if ~isfield(cfg.LabelPointsPlot,'TextProps')
        FLDS = {'FontSize'};
        cfg.LabelPointsPlot.TextProps = structrestrictflds(cfg.LabelPointsPlot,FLDS);
        cfg.LabelPointsPlot = rmfield(cfg.LabelPointsPlot,FLDS);
        
        cfg.LabelPointsPlot.TextOffset = cfg.LabelPointsPlot.LblOffset;
        cfg.LabelPointsPlot = rmfield(cfg.LabelPointsPlot,'LblOffset');
      end
      
      if ~isfield(cfg.Track.PredictPointsPlot,'MarkerProps')
        cfg.Track.PredictPointsPlot = struct('MarkerProps',cfg.Track.PredictPointsPlot);
      end        
      if isfield(cfg.Track,'PredictPointsPlotColorMapName')
        cfg.Track.PredictPointsPlot.ColorMapName = cfg.Track.PredictPointsPlotColorMapName;
        cfg.Track = rmfield(cfg.Track,'PredictPointsPlotColorMapName');
      end
      if isfield(cfg.Track,'PredictPointsPlotColors')
        cfg.Track.PredictPointsPlot.Colors = cfg.Track.PredictPointsPlotColors;
        cfg.Track = rmfield(cfg.Track,'PredictPointsPlotColors');
      end
      if isfield(cfg.Track,'PredictPointsShowTextLbl')
        cfg.Track.PredictPointsPlot.TextProps.Visible = ...
          onIff(cfg.Track.PredictPointsShowTextLbl);
        cfg.Track = rmfield(cfg.Track,'PredictPointsShowTextLbl');
      end
    end    
    function cfg = cfgModernize(cfg)
      % Bring a cfg up-to-date with latest by adding in any new fields from
      % config.default.yaml.

      cfg = Labeler.cfgModernizeBase(cfg);
      
      cfgBase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      
      cfg = structoverlay(cfgBase,cfg,'dontWarnUnrecog',true,...
        'allowedUnrecogFlds',{'Colors'});% 'ColorsSets'});
      view = augmentOrTruncateVector(cfg.View,cfg.NumViews);
      cfg.View = view(:);
    end
    
    function cfg = cfgDefaultOrder(cfg)
      % Reorder fields of cfg struct to default order
      
      cfg0 = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      flds0 = fieldnames(cfg0);
      flds = fieldnames(cfg);
      flds0 = flds0(ismember(flds0,flds)); 
      fldsExtra = setdiff(flds,flds0);
      cfg = orderfields(cfg,[flds0(:);fldsExtra(:)]);
    end
    
    function cfg = cfgRmLabelPoints(cfg,iptsrm)
      % Update/Massage config, removing given landmarks/pts 
      %
      % iptsrm: vector of ipt indices

      nptsrm = numel(iptsrm);
      nptsremain = cfg.NumLabelPoints-nptsrm;
      if nptsremain<=0
        error('Cannot remove %d points as config.NumLabelPoints is %d.',...
          nptsrm,cfg.NumLabelPoints);
      end
      
      cfg.LabelPointNames(iptsrm,:) = [];
      cfg.NumLabelPoints = nptsremain;
    end
    
    function cfg = cfgAddLabelPoints(cfg,nptsadd)
      % Update/Massage config, adding nptsadd new landmarks/pts 
      %
      % nptsadd: scalar positive int
      
      npts0 = cfg.NumLabelPoints;
      newptnames = arrayfun(@(x)sprintf('pt%d',x),...
        (npts0+1:npts0+nptsadd)','uni',0);

      cfg.LabelPointNames = cat(1,cfg.LabelPointNames,newptnames);
      cfg.NumLabelPoints = cfg.NumLabelPoints + nptsadd;
    end      
    
    % moved this from ProjectSetup
    function sMirror = cfg2mirror(cfg)
      % Convert true/full data struct to 'mirror' struct for adv table. (The term
      % 'mirror' comes from implementation detail of adv table/propertiesGUI.)
      
      nViews = cfg.NumViews;
      nPoints = cfg.NumLabelPoints;
      
      cfg = Labeler.cfgDefaultOrder(cfg);
      sMirror = rmfield(cfg,{'NumViews' 'NumLabelPoints'}); % 'LabelMode'});
      sMirror.Track = rmfield(sMirror.Track,{'Enable' 'Type'});
      
      assert(isempty(sMirror.ViewNames) || numel(sMirror.ViewNames)==nViews);
      flds = arrayfun(@(i)sprintf('view%d',i),(1:nViews)','uni',0);
      if isempty(sMirror.ViewNames)
        vals = repmat({''},nViews,1);
      else
        vals = sMirror.ViewNames(:);
      end
      sMirror.ViewNames = cell2struct(vals,flds,1);
      
      assert(isempty(sMirror.LabelPointNames) || numel(sMirror.LabelPointNames)==nPoints);
      flds = arrayfun(@(i)sprintf('point%d',i),(1:nPoints)','uni',0);
      if isempty(sMirror.LabelPointNames)
        vals = repmat({''},nPoints,1);
      else
        vals = sMirror.LabelPointNames;
      end
      sMirror.LabelPointNames = cell2struct(vals,flds,1);
    end
    
    % moved this from ProjectSetup
    function sMirror = hlpAugmentOrTruncNameField(sMirror,fld,subfld,n)
      v = sMirror.(fld);
      flds = fieldnames(v);
      nflds = numel(flds);
      if nflds>n
        v = rmfield(v,flds(n+1:end));
      elseif nflds<n
        for i=nflds+1:n
          v.([subfld num2str(i)]) = '';
        end
      end
      sMirror.(fld) = v;
    end
    
    function sMirror = hlpAugmentOrTruncStructField(sMirror,fld,n)

      v = sMirror.(fld);
      v = augmentOrTruncateVector(v,n);
      sMirror.(fld) = v(:);
      
    end
        
  end
  
  %% Project/Lbl files
  methods
   
    function projNew(obj,name)
      % Create new project based on current configuration
      
      if exist('name','var')==0
        resp = inputdlg('Project name:','New Project');
        if isempty(resp)
          return;
        end
        name = resp{1};
      end
      % AL empty projnames can cause trouble lets just set a default now if 
      % nec
      if isempty(name)
        name = 'APTproject';
      end

      obj.isinit = true;

      obj.projname = name;
      obj.projFSInfo = [];
      obj.projGetEnsureTempDir('cleartmp',true);
      obj.movieFilesAll = cell(0,obj.nview);
      obj.movieFilesAllGT = cell(0,obj.nview);
      obj.movieFilesAllHaveLbls = zeros(0,1);
      obj.movieFilesAllGTHaveLbls = zeros(0,1);
      obj.movieInfoAll = cell(0,obj.nview);
      obj.movieInfoAllGT = cell(0,obj.nview);
      obj.movieFilesAllCropInfo = cell(0,1);
      obj.movieFilesAllGTCropInfo = cell(0,1);
      obj.movieFilesAllHistEqLUT = cell(0,obj.nview);
      obj.movieFilesAllGTHistEqLUT = cell(0,obj.nview);
      obj.cropIsCropMode = false;
      obj.trxFilesAll = cell(0,obj.nview);
      obj.trxFilesAllGT = cell(0,obj.nview);
      obj.projMacros = struct();
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      obj.viewCalibrationDataGT = [];
      obj.labelTemplate = []; % order important here
      obj.movieSetNoMovie(); % order important here
      obj.labeledpos = cell(0,1);
      obj.labeledposGT = cell(0,1);
      obj.labeledposTS = cell(0,1);
      obj.lastLabelChangeTS = 0;
      obj.labeledposTSGT = cell(0,1);
      obj.labeledposMarked = cell(0,1);
      obj.labeledpostag = cell(0,1);
      obj.labeledpostagGT = cell(0,1);
      obj.labeledpos2 = cell(0,1);
      obj.labeledpos2GT = cell(0,1);
      obj.gtIsGTMode = false;
      obj.gtSuggMFTable = MFTable.emptyTable(MFTable.FLDSID);
      obj.gtSuggMFTableLbled = false(0,1);
      obj.gtTblRes = [];
      
      obj.trackResInit();
      
      obj.isinit = false;
      
      obj.updateFrameTableComplete();  
      obj.labeledposNeedsSave = false;
      obj.needsSave = false;

      trkPrefs = obj.projPrefs.Track;
      if trkPrefs.Enable
        % Create default trackers
        assert(isempty(obj.trackersAll));
        trkersCreateInfo = LabelTracker.getAllTrackersCreateInfo;
        nTrkers = numel(trkersCreateInfo);
        tAll = cell(1,nTrkers);
        for i=1:nTrkers
          trkerCls = trkersCreateInfo{i}{1};
          trkerClsArgs = trkersCreateInfo{i}(2:end);
          tAll{i} = feval(trkerCls,obj,trkerClsArgs{:});
          tAll{i}.init();
        end
        obj.trackersAll = tAll;
        obj.currTracker = 1;
      else
        obj.currTracker = 0;
      end

      props = obj.gdata.propsNeedInit;
      for p = props(:)', p=p{1}; %#ok<FXSET>
        obj.(p) = obj.(p);
      end
%       obj.setShowPredTxtLbl(obj.showPredTxtLbl);
      
      obj.notify('cropIsCropModeChanged');
      obj.notify('gtIsGTModeChanged');
    end
      
    function projSaveRaw(obj,fname)
      s = obj.projGetSaveStruct();
      
      if 1
        try
          rawLblFile = obj.projGetRawLblFile();
          save(rawLblFile,'-mat','-struct','s');
          obj.projBundleSave(fname);
        catch ME
          save(fname,'-mat','-struct','s');
          msg = ME.getReport();
          warningNoTrace('Saved raw project file %s. Error caught during bundled project save: %s\n',...
            fname,msg);          
        end
      else
        save(fname,'-mat','-struct','s');
      end
      obj.labeledposNeedsSave = false;
      obj.needsSave = false;
      obj.projFSInfo = ProjectFSInfo('saved',fname);

      RC.saveprop('lastLblFile',fname);      
    end
    
    function projSaveModified(obj,fname,varargin)
      s = obj.projGetSaveStructWithMassage(varargin{:});
      save(fname,'-mat','-struct','s');
      fprintf('Saved modified project file %s.\n',fname);
    end
        
    function [success,lblfname] = projSaveAs(obj,lblfname)
      % Saves a .lbl file, prompting user for filename.

      if nargin <= 1,
        if ~isempty(obj.projectfile)
          filterspec = obj.projectfile;
        else
          % Guess a path/location for save
          lastLblFile = RC.getprop('lastLblFile');
          if isempty(lastLblFile)
            if obj.hasMovie
              savepath = fileparts(obj.moviefile);
            else
              savepath = pwd;
            end
          else
            savepath = fileparts(lastLblFile);
          end
          
          if ~isempty(obj.projname)
            projfile = sprintf(obj.DEFAULT_LBLFILENAME,obj.projname);
          else
            projfile = sprintf(obj.DEFAULT_LBLFILENAME,'APTProject');
          end
          filterspec = fullfile(savepath,projfile);
        end
        
        [lblfname,pth] = uiputfile(filterspec,'Save label file');
        if isequal(lblfname,0)
          lblfname = [];
          success = false;
          return;
        end
        lblfname = fullfile(pth,lblfname);
      end

      success = true;
      obj.projSaveRaw(lblfname);

    end
    
    function [success,lblfname] = projSaveSmart(obj)
      % Try to save to current project; if there is no project, do a saveas
      lblfname = obj.projectfile;
      if isempty(lblfname)
        [success,lblfname] = obj.projSaveAs();
      else
        success = true;
        obj.projSaveRaw(lblfname);
      end
    end
    
    function s = projGetSaveStruct(obj,varargin)
      % Warning: if .preProcSaveData is true, then s.preProcData is a
      % handle (shallow copy) to obj.preProcData
      
      [sparsify,forceIncDataCache,forceExcDataCache,macroreplace,...
        savepropsonly,massageCropProps] = ...
        myparse(varargin,...
        'sparsify',true,...
        'forceIncDataCache',false,... % include .preProcData* and .ppdb even if .preProcSaveData is false
        'forceExcDataCache',false, ... 
        'macroreplace',false, ... % if true, use mfaFull/tfaFull for mfa/tfa
        'savepropsonly',false, ... % if true, just get the .SAVEPROPS with no further massage
        'massageCropProps',false ... % if true, structize crop props
        );
      assert(~(forceExcDataCache&&forceIncDataCache));      

      s = struct();
      s.cfg = obj.getCurrentConfig();
      
      if sparsify
        lposProps = obj.SAVEPROPS_LPOS(:,1);
        lposPropsType = obj.SAVEPROPS_LPOS(:,2);
        
        for f=obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
          iLpos = find(strcmp(f,lposProps));
          if isempty(iLpos)
            s.(f) = obj.(f);
          else
            lpostype = lposPropsType{iLpos};
            xarrFull = obj.(f);
            assert(iscell(xarrFull));
            xarrSprs = cellfun(@(x)SparseLabelArray.create(x,lpostype),...
              xarrFull,'uni',0);
            s.(f) = xarrSprs;
          end
        end
      else
        for f=obj.SAVEPROPS, f=f{1}; %#ok<FXSET>
          s.(f) = obj.(f);
        end
      end
      
      if macroreplace
        s.movieFilesAll = obj.movieFilesAllFull;
        s.movieFilesAllGT = obj.movieFilesAllGTFull;
        s.trxFilesAll = obj.trxFilesAllFull;
        s.trxFilesAllGT = obj.trxFilesAllGTFull;
      end
      if massageCropProps
        cellOfObjArrs2CellOfStructArrs = ...
          @(x)cellfun(@(y)arrayfun(@struct,y),x,'uni',0); % note, y can be []
        warnst = warning('off','MATLAB:structOnObject');
        s.movieFilesAllCropInfo = cellOfObjArrs2CellOfStructArrs(obj.movieFilesAllCropInfo);
        s.movieFilesAllGTCropInfo = cellOfObjArrs2CellOfStructArrs(obj.movieFilesAllGTCropInfo);
        warning(warnst);
        s.cropProjHasCrops = obj.cropProjHasCrops;
      end
      if savepropsonly
        return
      end
      
      % Older comment: clean information we shouldn't save from AWS EC2
      % AL 20191217: setting .awsec2 to [] breaks training, trackDLBackEnd 
      % is a handle to a live object. 
      %
      % We do a deep-copy of the backend here as we are serializing the 
      % proj either for saving or a stripped lbl etc and i) the saved 
      % objects need sanitation and ii) conceptually the serialized object
      % does not share handle identity with other 'live' handles to obj.
      if isfield(s,'trackDLBackEnd') && ~isempty(s.trackDLBackEnd)
        s.trackDLBackEnd = s.trackDLBackEnd.copyAndDetach();
      end
      
      switch obj.labelMode
        case LabelMode.TEMPLATE
          %s.labelTemplate = obj.lblCore.getTemplate();
      end

      tObjAll = obj.trackersAll;
      s.trackerClass = cellfun(@getTrackerClassAugmented,tObjAll,'uni',0);
      s.trackerData = cellfun(@getSaveToken,tObjAll,'uni',0);
      
      if ~forceExcDataCache && ( obj.preProcSaveData || forceIncDataCache )
        s.preProcData = obj.preProcData; % Warning: shallow copy for now, caller should not mutate
        s.preProcDataTS = obj.preProcDataTS;
        s.ppdb = obj.ppdb;
      end
    end
    
    function s = projGetSaveStructWithMassage(obj,varargin)
      [massageType,massageArg] = myparse(varargin,...
        'modificationType','addpoints',... % or 'rmpoints'
        'modificationArg',1 ...
      );
      
      s = obj.projGetSaveStruct('sparsify',false,'forceExcDataCache',true);
      
      switch massageType
        case 'addpoints'
          nptsadd = massageArg;
          fprintf(1,'Adding %d points to project...\n',nptsadd);
          s.cfg = Labeler.cfgAddLabelPoints(s.cfg,nptsadd);
          %modfcn will be applied to each el of lposProps
          modfcn = @(x,ty)SparseLabelArray.fullExpandPts(x,ty,nptsadd);          
        case 'rmpoints'
          iptsrm = massageArg;
          fprintf(1,'Removing points %s from project...\n',mat2str(iptsrm));
          s.cfg = Labeler.cfgRmLabelPoints(s.cfg,iptsrm);
          modfcn = @(x,ty)SparseLabelArray.fullRmPts(x,iptsrm);
        otherwise
          assert(false);
      end
      
      % massage/sparsify lpos props
      lposProps = obj.SAVEPROPS_LPOS;
      nprop = size(lposProps,1);
      for iprop=1:nprop
        fld = lposProps{iprop,1};
        ty = lposProps{iprop,2};
        val = s.(fld);
        switch fld
          case {'labeledpos2' 'labeledpos2GT'}
            % Clear imported tracking; below we clear trackers
            for imov=1:numel(val)
              val{imov}(:) = nan;
            end
        end
        val = cellfun(@(x)modfcn(x,ty),val,'uni',0);
        s.(fld) = cellfun(@(x)SparseLabelArray.create(x,ty),val,'uni',0);
      end

      % manual massage other fields of s
      % *Warning* Not very maintainable, not very happy but this may not 
      % get used that much plus it should typically be non-critical (eg if 
      % an error occurs, the original proj is safe/untouched).
      s.gtTblRes = [];
      s.labelTemplate = [];
      if ~isempty(s.trackParams) && ...
         ~strcmp(s.trackParams.ROOT.CPR.RotCorrection.OrientationType,'fixed')
        warningNoTrace('CPR rotational correction/orientation type is not ''fixed''. Head/tail landmarks updated to landmarks 1/2 respectively.');
        s.trackParams.ROOT.CPR.RotCorrection.HeadPoint = 1;
        s.trackParams.ROOT.CPR.RotCorrection.TailPoint = 2;
      end
      s.xvResults = [];
      s.xvResultsTS = [];
      s.skeletonEdges = zeros(0,2);
      s.flipLandmarkMatches = zeros(0,2);
      s = Labeler.resetTrkResFieldsStruct(s);
      for i=1:numel(s.trackerData)
        s.trackerData{i} = [];
      end
    end
    
    function currMovInfo = projLoad(obj,fname,varargin)
      % Load a lbl file
      %
      % currProjInfo: scalar struct containing diagnostic info. When the
      % project is loaded, APT attempts to set the movie to the last-known
      % (saved) movie. If this movie is not found in the filesys, the
      % project will be set to "nomovie" and currProjInfo will contain:
      %   .iMov: movie(set) index for desired-but-not-found movie
      %   .badfile: path of file-not-found
      %
      % If the movie is able to set the project correctly, currProjInfo
      % will be [].
            
      nomovie = myparse(varargin,...
        'nomovie',false ... % If true, call movieSetNoMovie() instead of movieSet(currMovie)
        );
            
      currMovInfo = [];
      
      if exist('fname','var')==0
        lastLblFile = RC.getprop('lastLblFile');
        if isempty(lastLblFile)
          lastLblFile = pwd;
        end
        filterspec = sprintf(obj.DEFAULT_LBLFILENAME,'*');
        [fname,pth] = uigetfile(filterspec,'Load label file',lastLblFile);
        if isequal(fname,0)
          return;
        end
        fname = fullfile(pth,fname);
      end
      
      if exist(fname,'file')==0
        error('Labeler:file','File ''%s'' not found.',fname);
      else
        tmp = which(fname);
        if ~isempty(tmp)
          fname = tmp; % use fullname
        else
          % fname exists, but cannot be expanded into a fullpath; ideally 
          % the only possibility is that it is already a fullpath
        end
      end
      
      % MK 20190204. Use Unbundling instead of loading.
      [success,tlbl,wasbundled] = obj.projUnbundleLoad(fname);
      if ~success, error('Could not unbundle the label file %s',fname); end
      
      % AL 20191002 occlusion-prediction viz, DLNetType enum changed
      % should-be-harmlessly
      warnst = warning('off','MATLAB:class:EnumerationValueChanged');
      s = load(tlbl,'-mat');
      warning(warnst);
%       s = load(fname,'-mat');  

      if ~all(isfield(s,{'VERSION' 'labeledpos'}))
        error('Labeler:load','Unexpected contents in Label file.');
      end
      RC.saveprop('lastLblFile',fname);

      s = Labeler.lblModernize(s);
      
      obj.isinit = true;

      obj.initFromConfig(s.cfg);
      
      % From here to the end of this method is a parallel initialization to
      % projNew()
      
      LOADPROPS = Labeler.SAVEPROPS(~ismember(Labeler.SAVEPROPS,...
                                              Labeler.SAVEBUTNOTLOADPROPS));
      lposProps = obj.SAVEPROPS_LPOS(:,1);
      for f=LOADPROPS(:)',f=f{1}; %#ok<FXSET>
        if isfield(s,f)          
          if ~any(strcmp(f,lposProps))
            obj.(f) = s.(f);
          else
            val = s.(f);
            assert(iscell(val));
            for iMov=1:numel(val)
              x = val{iMov};
              if isstruct(x)
                xfull = SparseLabelArray.full(x);
                val{iMov} = xfull;
              end
            end
            obj.(f) = val;
          end
        else
          warningNoTrace('Labeler:load','Missing load field ''%s''.',f);
          %obj.(f) = [];
        end
      end
     
      obj.computeLastLabelChangeTS();
      %fcnAnyNonNan = @(x)any(~isnan(x(:)));
      fcnNumLbledRows = @Labeler.computeNumLbledRows;
      obj.movieFilesAllHaveLbls = cellfun(fcnNumLbledRows,obj.labeledpos);
      obj.movieFilesAllGTHaveLbls = cellfun(fcnNumLbledRows,obj.labeledposGT);      
      obj.gtUpdateSuggMFTableLbledComplete();
      
      % need this before setting movie so that .projectroot exists
      obj.projFSInfo = ProjectFSInfo('loaded',fname);

      % Tracker.
      nTracker = numel(s.trackerData);
      assert(nTracker==numel(s.trackerClass));
      assert(isempty(obj.trackersAll));
      tAll = cell(1,nTracker);
      for i=1:nTracker 
        tAll{i} = LabelTracker.create(obj,s.trackerClass{i},s.trackerData{i});
      end
      obj.trackersAll = tAll;
      
      obj.isinit = false;

      % preproc data cache
      % s.preProcData* will be present iff s.preProcSaveData==true
      if s.preProcSaveData 
        if isempty(s.preProcData)
          assert(obj.preProcData.N==0);
        else
          fprintf('Loading data cache: %d rows.\n',s.preProcData.N);
          obj.preProcData = s.preProcData;
          obj.preProcDataTS = s.preProcDataTS;
        end
        if isempty(s.ppdb)
          assert(obj.ppdb.dat.N==0);
        else
          fprintf('Loading DL data cache: %d rows.\n',s.ppdb.dat.N);
          obj.ppdb = s.ppdb;
        end
      end

      if obj.nmoviesGTaware==0 || s.currMovie==0 || nomovie
        obj.movieSetNoMovie();
      else
        [tfok,badfile] = obj.movieCheckFilesExistSimple(s.currMovie,s.gtIsGTMode);
        if ~tfok
          currMovInfo.iMov = s.currMovie;
          currMovInfo.badfile = badfile;
          obj.movieSetNoMovie();
        else
          obj.movieSet(s.currMovie);
          [tfok] = obj.checkFrameAndTargetInBounds(s.currFrame,s.currTarget);
          if ~tfok,
            warning('Cached frame number %d and target number %d are out of bounds for movie %d, reverting to using first frame of first target.',s.currFrame,s.currTarget,s.currMovie);
            s.currFrame = 1;
            s.currTarget = 1;
          end
          obj.setFrameAndTarget(s.currFrame,s.currTarget);
        end
      end
      
%       % Needs to occur after tracker has been set up so that labelCore can
%       % communicate with tracker if necessary (in particular, Template Mode 
%       % <-> Hide Predictions)
%       obj.labelingInit();

      obj.labeledposNeedsSave = false;
      obj.needsSave = false;
%       obj.suspScore = obj.suspScore;
            
      obj.updateFrameTableComplete(); % TODO don't like this, maybe move to UI
      
      if obj.currMovie>0
        obj.labelsUpdateNewFrame(true);
      end
      
      % This needs to occur after .labeledpos etc has been set
      pamode = PrevAxesMode.(s.cfg.PrevAxes.Mode);
      [~,prevModeInfo] = obj.FixPrevModeInfo(pamode,s.cfg.PrevAxes.ModeInfo);      
      obj.setPrevAxesMode(pamode,prevModeInfo);
      
      % Call this here to eg init AWS backend
      obj.trackSetDLBackend(obj.trackDLBackEnd);
      
      props = obj.gdata.propsNeedInit;
      for p = props(:)', p=p{1}; %#ok<FXSET>
        obj.(p) = obj.(p);
      end
      
      obj.setSkeletonEdges(obj.skeletonEdges);
      obj.setShowSkeleton(obj.showSkeleton);
      obj.setFlipLandmarkMatches(obj.flipLandmarkMatches);
%       obj.setShowPredTxtLbl(obj.showPredTxtLbl);
      
      if ~wasbundled
        % DMC.rootDir point to original model locs
        % Immediately modernize model wrt dl-cache-related invariants
        % After this branch the model is as good as a bundled load
        
        fprintf(1,'\n\n### Raw/unbundled project migration.\n');
        fprintf(1,'Copying Deep Models into %s.\n',obj.projTempDir);
        for iTrker = 1:numel(obj.trackersAll)
          tObj = obj.trackersAll{iTrker};
          if isprop(tObj,'trnLastDMC') && ~isempty(tObj.trnLastDMC)            
            dmc = tObj.trnLastDMC;            
            for ivw = 1:numel(dmc)
              dm = dmc(ivw);
              try                
                if dm.isRemote
                  warningNoTrace('Net %s, view %d. Remote Model detected. This will not migrated/preserved.',dm.netType,ivw);
                  continue;
                end
                
                if ivw==1
                  fprintf(1,'Detected model for nettype ''%s'' in %s.\n',...
                    dm.netType,dm.rootDir);
                end
                
                tfsucc = dm.updateCurrInfo();
                if ~tfsucc
                  warningNoTrace('Failed to update model iteration for model with net type %s.',...
                    char(dm.netType));
                end
                
                modelFiles = dm.findModelGlobsLocal();
                assert(~strcmp(dm.rootDir,obj.projTempDir)); % Possible filesep issues
                modelFilesDst = strrep(modelFiles,dm.rootDir,obj.projTempDir);
                for mndx = 1:numel(modelFiles)
                  copyfileensuredir(modelFiles{mndx},modelFilesDst{mndx}); % throws
                  % for a given tracker, multiple DMCs this could re-copy
                  % proj-level artifacts like stripped lbls
                  fprintf(1,'%s -> %s\n',modelFiles{mndx},modelFilesDst{mndx});
                end                
              catch ME
                warningNoTrace('Nettype ''%s'' (view %d): error caught trying to save model. Trained model will not be migrated for this net type:\n%s',...
                  dm.netType,ivw,ME.getReport());
              end
            end
          end
        end
      end
      fprintf(1,'\n\n');
      obj.projUpdateDLCache(); % this can fail (output arg not checked)
      
      obj.notify('projLoaded');
      obj.notify('cropUpdateCropGUITools');
      %obj.notify('cropIsCropModeChanged');
      obj.notify('gtIsGTModeChanged');
      obj.notify('gtSuggUpdated');
      obj.notify('gtResUpdated');
      
      fprintf('Finished loading project.\n');
      
    end
    
    function projImport(obj,fname)
      % 'Import' the project fname, MERGING movies/labels into the current project.
          
      assert(false,'Unsupported');
      
      if exist(fname,'file')==0
        error('Labeler:file','File ''%s'' not found.',fname);
      else
        tmp = which(fname);
        if ~isempty(tmp)
          fname = tmp; % use fullname
        else
          % fname exists, but cannot be expanded into a fullpath; ideally 
          % the only possibility is that it is already a fullpath
        end
      end
       
      [success, tlbl] = obj.projUnbundleLoad(fname);
      if ~success, error('Could not unbundle the label file %s',fname); end
      s = load(tlbl,'-mat');
      obj.projClearTempDir();
%       s = load(fname,'-mat');
      if s.nLabelPoints~=obj.nLabelPoints
        error('Labeler:projImport','Project %s uses nLabelPoints=%d instead of %d for the current project.',...
          fname,s.nLabelPoints,obj.nLabelPoints);
      end
      
      assert(~obj.isMultiView && iscolumn(s.movieFilesAll));
      
      if isfield(s,'projMacros') && ~isfield(s.projMacros,'projdir')
        s.projMacros.projdir = fileparts(fname);
      else
        s.projMacros = struct();
      end
      
      nMov = size(s.movieFilesAll,1);
      for iMov = 1:nMov
        movfile = s.movieFilesAll{iMov,1};
        movfileFull = Labeler.platformize(FSPath.macroReplace(movfile,s.projMacros));
        movifo = s.movieInfoAll{iMov,1};
        trxfl = s.trxFilesAll{iMov,1};
        lpos = s.labeledpos{iMov};
        lposTS = s.labeledposTS{iMov};
        lpostag = s.labeledpostag{iMov};
        if isempty(s.suspScore)
          suspscr = [];
        else
          suspscr = s.suspScore{iMov};
        end
        
        if exist(movfileFull,'file')==0 || ~isempty(trxfl)&&exist(trxfl,'file')==0
          warning('Labeler:projImport',...
            'Missing movie/trxfile for movie ''%s''. Not importing this movie.',...
            movfileFull);
          continue;
        end
           
        obj.movieFilesAll{end+1,1} = movfileFull;
        obj.movieFilesAllHaveLbls(end+1,1) = any(~isnan(lpos(:)));
        obj.movieInfoAll{end+1,1} = movifo;
        obj.trxFilesAll{end+1,1} = trxfl;
        obj.labeledpos{end+1,1} = lpos;
        obj.labeledposTS{end+1,1} = lposTS;
        obj.lastLabelChangeTS = max(obj.lastLabelChangeTS,max(lposTS(:)));
        obj.labeledposMarked{end+1,1} = false(size(lposTS));
        obj.labeledpostag{end+1,1} = lpostag;
        obj.labeledpos2{end+1,1} = s.labeledpos2{iMov};
        if ~isempty(obj.suspScore)
          obj.suspScore{end+1,1} = suspscr;
        end
%         if ~isempty(obj.suspNotes)
%           obj.suspNotes{end+1,1} = [];
%         end
      end

      obj.labeledposNeedsSave = true;
      obj.projFSInfo = ProjectFSInfo('imported',fname);
      
      % TODO prob would need .preProcInit() here
      
      if ~isempty(obj.tracker)
        warning('Labeler:projImport','Re-initting tracker.');
        obj.tracker.init();
      end
      % TODO .trackerDeep
    end
    
    function projAssignProjNameFromProjFileIfAppropriate(obj)
      if isempty(obj.projname) && ~isempty(obj.projectfile)
        [~,fnameS] = fileparts(obj.projectfile);
        obj.projname = fnameS;
      end
    end
    
    % Functions to handle bundled label files
    % MK 20190201
    function tname = projGetEnsureTempDir(obj,varargin) % throws
      % tname: project tempdir, assigned to .projTempDir. Guaranteed to
      % exist, contents not guaranteed
      
      cleartmp = myparse(varargin,...
        'cleartmp',false...
        );
      
      if isempty(obj.projTempDir)
        obj.projTempDir = tempname(APT.getdlcacheroot);
      end
      tname = obj.projTempDir;
      
      if exist(tname,'dir')==0
        [success,message,~] = mkdir(tname);
        if ~success
          error('Could not create temp directory %s: %s',tname,message);
        end        
      elseif cleartmp
        obj.projClearTempDir();
      else
        % .projTempDir exists and may have stuff in it
      end
      
    end
    
    function [success,rawLblFile,isbundled] = projUnbundleLoad(obj,fname) % throws
      % Unbundles the lbl file if it is a tar bundle.
      %
      % This can throw. If it throws you know nothing.
      %
      % If it doesn't throw:
      %  If success and isbundled, then .projTempDir/<rawLblFile> is the
      %   raw label file and .projTempDir/ contains the exploded DL model
      %   cache tree.
      %  If success and ~isbundled, then .projTempDir/<rawLblFile> is the
      %   raw label file, but there are no DL models under .projTempDir.
      %  If ~success, then .projTempDir is set and exists on the filesystem
      %   but that's it. .projTempDir/<rawLblFile> probably doesn't exist.
      %   
      % fname: fullpath to projectfile, either tarred/bundled or untarred
      %
      % success, isbundled: as described above
      % rawLblFile: full path to untarred raw label file in .projTempDir
      % MK 20190201
      
      [rawLblFile,tname] = obj.projGetRawLblFile('cleartmp',true); % throws; sets .projTempDir
      
      try
        fprintf('Untarring project into %s\n',tname);
        untar(fname,tname);
        obj.unTarLoc = tname;
        fprintf('... done with untar.\n');
      catch ME
        if endsWith(ME.identifier,'invalidTarFile')
          warningNoTrace('Label file %s is not bundled. Using it in raw (mat) format.',fname);
          [success,message,~] = copyfile(fname,rawLblFile);
          if ~success
            warningNoTrace('Could not copy raw label file: %s',message);
            isbundled = [];
          else
            isbundled = false;  
            obj.unTarLoc = tname;
          end          
          return;
        else
          ME.rethrow(); % most unfortunate
        end
      end
      
      if ~exist(rawLblFile,'file')
        warning('Could not find raw label file in the bundled label file %s',fname);
        success = false;
        isbundled = [];
        return;
      end
      
      success = true;
      isbundled = true;
    end
    
    function success = cleanUpProjTempDir(obj,verbose)
      
      success = false;
      if nargin < 2,
        verbose = true;
      end
      if ~ischar(obj.unTarLoc) || isempty(obj.unTarLoc),
        success = true;
        return;
      end
      if ~exist(obj.unTarLoc,'dir'),
        if verbose,
          fprintf('Temporary tar directory %s does not exist. Not cleaning.\n',obj.unTarLoc);
        end
        return;
      end
      [success,msg] = rmdir(obj.unTarLoc,'s');
      if ~success && verbose,
        fprintf('Error deleting temporary tar directory %s:\n%s\n',obj.unTarLoc,msg);
      end
      if success,
        if verbose,
          fprintf('Removed temporary tar directory %s.\n',obj.unTarLoc);
        end
        obj.unTarLoc = '';
      end
      
    end
    
    function success = projUpdateDLCache(obj)
      % Updates project DL state to point to new cache in .projTempDir
      % 
      % Preconds: 
      %   - .projTempDir must be set
      %   - bundled project untarred into .projTempDir
      %   - .trackersAll has been set/loaded, ie 
      %       .trackersAll{iDLTrker}.trnLastDMC(ivw) is configured for
      %       running except possibly for .rootDir specification. 
      %
      % Postconds, success:
      %   - .trackersAll{iDLTrker}.trnLastDMC(ivw).rootDir updated to point
      %   to .projTempDir 
      %   - any memory of tracking results in DL tracker objs cleared
      %
      % Postcond, ~success: nothing changed      
      
      success = false;
      
      cacheDir = obj.projTempDir;
      
      % Check for exploded cache in tempdir      
      tCacheDir = fullfile(cacheDir,obj.projname);
      if ~exist(tCacheDir,'dir')
        warningNoTrace('Could not find model data for %s in temp directory %s. Deep Learning trackers not restored.',...
          obj.projname,cacheDir);
        return;
      end
            
      % Update/set all DMC.rootDirs to cacheDir
      tAll = obj.trackersAll;
      for iTrker = 1:numel(tAll)
        tObj = tAll{iTrker};
        if isa(tObj,'DeepTracker')
          dmc = tObj.trnLastDMC;
          for ivw = 1:numel(dmc)
            if ~dmc(ivw).isRemote
              dmc(ivw).rootDir = cacheDir;
              
              % This was a mistake as APT#2 could clear the partial 
              % trkfiles used by APT#1
%               mcDir = dmc(ivw).dirModelChainLnx;
%               if exist(mcDir,'dir')>0
%                 fprintf(1,'Cleaning %s\n',mcDir);
%                 [success,message] = rmdir(mcDir,'s');
%                 if ~success
%                   warningNoTrace('Could not clean local modelChain cache dir %s:',mcDir);
%                   warningNoTrace(message);
%                   
%                   % Nonfatal dont return
%                 end
%               end
            else
              warningNoTrace('Unexpected remote DMC detected for net %s, view %d.',...
                tObj.trnNetType.prettyStr,ivw);
              % At save-time we should be updating DMCs to local
              
              % Don't update dmc(ivw).rootDir in this case
              
              % Nonfatal dont return
            end
          end
          
          % Don't retain any previous tracking results
          tObj.clearTrackingResults();
        end
      end
      
      %obj.trackParams.ROOT.DeepTrack.Saving.CacheDir = cacheDir;
      
      success = true;
    end
    
    function [rawLblFile,projtempdir] = projGetRawLblFile(obj,varargin) % throws
      projtempdir = obj.projGetEnsureTempDir(varargin{:});
      rawLblFile = fullfile(projtempdir,obj.DEFAULT_RAW_LABEL_FILENAME);
    end
    
    function projBundleSave(obj,outFile,varargin) % throws 
      % bundle contents of projTempDir into outFile
      %
      % throws on err, hopefully cleans up after itself (projtempdir) 
      % regardless. 
      
      verbose = myparse(varargin,...
        'verbose',obj.projVerbose ...
      );
      
      [rawLblFile,projtempdir] = obj.projGetRawLblFile();
      if ~exist(rawLblFile,'file')
        error('Raw label file %s does not exist. Could not create bundled label file.',...
          rawLblFile);
      end
      
      % allModelFiles will contain all projtempdir artifacts to be tarred
      allModelFiles = {rawLblFile};
      
      % find the model files and then bundle them into the tar directory.
      % but since there isn't much in way of relative path support in
      % matlabs tar/zip functions, we will also have to copy them first the
      % temp directory. sigh.
      
      for iTrker = 1:numel(obj.trackersAll)
        tObj = obj.trackersAll{iTrker};
        if isprop(tObj,'trnLastDMC') && ~isempty(tObj.trnLastDMC)
          % a lot of unnecessary moving around is to maintain the directory
          % structure - MK 20190204
          
          dmc = tObj.trnLastDMC;
        
          for ndx = 1:numel(dmc)
            dm = dmc(ndx);
            
            try
              tfsucc = dm.updateCurrInfo();
              if ~tfsucc
                warningNoTrace('Failed to update model iteration for model with net type %s.',...
                  char(dm.netType));
              end

              if dm.isRemote
                try
                  dm.mirrorFromRemoteAws(projtempdir);
                catch
                  warningNoTrace('Could not check if trackers had been downloaded from AWS.');
                end
              end

              if verbose>0 && ndx==1
                fprintf(1,'Saving model for nettype ''%s'' from %s.\n',...
                  dm.netType,dm.rootDir);
              end

              modelFiles = dm.findModelGlobsLocal();
              if strcmp(dm.rootDir,projtempdir) % Possible filesep issues 
                % DMC already lives in the right place
                if verbose>1
                  cellfun(@(x)fprintf(1,'%s\n',x),modelFiles);
                end
                modelFilesDst = modelFiles;
              else
                % eg legacy projects (raw/unbundled)
                modelFilesDst = strrep(modelFiles,dm.rootDir,projtempdir);
                for mndx = 1:numel(modelFiles)
                  copyfileensuredir(modelFiles{mndx},modelFilesDst{mndx}); % throws
                  % for a given tracker, multiple DMCs this could re-copy
                  % proj-level artifacts like stripped lbls
                  if verbose>1                    
                    fprintf(1,'%s -> %s\n',modelFiles{mndx},modelFilesDst{mndx});
                  end
                end
              end
              allModelFiles = [allModelFiles; modelFilesDst(:)]; %#ok<AGROW>
            catch ME
              warningNoTrace('Nettype ''%s'' (view %d): error caught trying to save model. Trained model will not be saved for this net type:\n%s',...
                dm.netType,ndx,ME.getReport());
            end
          end
        end
      end
      
      % - all DL models exist under projtempdir
      % - obj...Saving.CacheDir is unchanged
      % - all DMCs need not have .rootDirs that point to projtempdir
                  
      pat = [regexprep(projtempdir,'\\','\\\\') '[/\\]'];
      allModelFiles = cellfun(@(x) regexprep(x,pat,''),...
        allModelFiles,'UniformOutput',false);
      fprintf(1,'Tarring %d model files into %s\n',numel(allModelFiles),projtempdir);
      tar([outFile '.tar'],allModelFiles,projtempdir);
      movefile([outFile '.tar'],outFile); 
      fprintf(1,'Project saved to %s\n',outFile);

      % matlab by default adds the .tar. So save it to tar
      % and then move it.
      
      % Don't clear the tempdir here, user may still be using project.
      %obj.clearTempDir();
    end
    
    function projExportTrainData(obj,outfile)
      
      [tfsucc,tblPTrn,s] = ...
        obj.trackCreateDeepTrackerStrippedLbl();
      if ~tfsucc,
        error('Could not collect data for exporting.');
      end
      % preProcData_P is [nLabels,nViews,nParts,2]
      save(outfile,'-mat','-v7.3','-struct','s');
      
    end
    
    function projClearTempDir(obj) % throws
      if isempty(obj.projTempDir)
        return;
      end
      
      [success, message, ~] = rmdir(obj.projTempDir,'s');
      if success
        fprintf(1,'Cleared temp dir: %s\n',obj.projTempDir);
      else
        warning('Could not clear the temp directory: %s',message);
      end
      [success, message, ~] = mkdir(obj.projTempDir);
      if ~success
        error('Could not clear the temp directory %s',message);
      end
    end
    
    function v = projDeterministicRandFcn(obj,randfcn)
      % Wrapper around rand() that sets/resets RNG seed
      s = rng(obj.projRngSeed);
      v = randfcn();
      rng(s);
    end
    
  end
  
  methods % projMacros
    
    % Macros defined in .projMacros are conceptually for movie files (or
    % "experiments", which are typically located at arbitrary locations in
    % the filesystem (which bear no relation to eg the location of the
    % project file).
    %
    % Other macros can be defined depending on the context. For instance,
    % trxFiles most typically are located in relation to their
    % corresponding movie and currently only a limited number of 
    % movie-related macros are supported. TrkFiles (tracking outputs) might 
    % be located in relation to their movies or perhaps their projects. etc
    
    function projMacroAdd(obj,macro,val)
      if isfield(obj.projMacros,macro)
        error('Labeler:macro','Macro ''%s'' already defined.',macro);
      end
      obj.projMacros.(macro) = val;
    end
    
    function projMacroSet(obj,macro,val)
      if ~ischar(val)
        error('Labeler:macro','Macro value must be a string.');
      end
      
      s = obj.projMacros;
      if ~isfield(s,macro)
        error('Labeler:macro','''%s'' is not a macro in this project.',macro);
      end
      s.(macro) = val;
      obj.projMacros = s;
    end
    
    function projMacroSetUI(obj)
      % Set any/all current macros with inputdlg
      
      s = obj.projMacros;
      macros = fieldnames(s);
      macrosdisp = cellfun(@(x)['$' x],macros,'uni',0);
      vals = struct2cell(s);
      nmacros = numel(macros);
      INPUTBOXWIDTH = 100;
      resp = inputdlgWithBrowse(macrosdisp,'Project macros',...
        repmat([1 INPUTBOXWIDTH],nmacros,1),vals);
      if ~isempty(resp)
        assert(isequal(numel(macros),numel(vals),numel(resp)));
        for i=1:numel(macros)
          try
            obj.projMacroSet(macros{i},resp{i});
          catch ME
            warningNoTrace('Labeler:macro','Cannot set macro ''%s'': %s',...
              macrosdisp{i},ME.message);
          end
        end
      end     
    end
    
    function s = projMacrosGetWithAuto(obj)
      % append auto-generated macros to .projMacros
      %
      % over time, many/most clients of .projMacros should probably call
      % this
      
      s = obj.projMacros;
      if ~isfield(s,'projdir') && ~isempty(obj.projectroot)
        % This conditional allows user to explictly specify project root
        % Useful use case here: testproject 'modules' (lbl + data in portable folder)
        s.projdir = obj.projectroot;
      end
    end
    
    function projMacroRm(obj,macro)
      if ~isfield(obj.projMacros,macro)
        error('Labeler:macro','Macro ''%s'' is not defined.',macro);
      end
      obj.projMacros = rmfield(obj.projMacros,macro);
    end
    
    function projMacroClear(obj)
      obj.projMacros = struct();
    end
    
    function projMacroClearUnused(obj)
      mAll = fieldnames(obj.projMacros);
      mfa = [obj.movieFilesAll(:); obj.movieFilesAllGT(:)];
      for macro=mAll(:)',macro=macro{1}; %#ok<FXSET>
        tfContainsMacro = cellfun(@(x)FSPath.hasMacro(x,macro),mfa);
        if ~any(tfContainsMacro)
          obj.projMacroRm(macro);
        end
      end
    end
    
    function tf = projMacroIsMacro(obj,macro)
      tf = isfield(obj.projMacros,macro);
    end
   
    function s = projMacroStrs(obj)
      m = obj.projMacrosGetWithAuto();
      flds = fieldnames(m);
      vals = struct2cell(m);
      s = cellfun(@(x,y)sprintf('%s -> %s',x,y),flds,vals,'uni',0);
    end    
    
    function p = projLocalizePath(obj,p)
      p = FSPath.fullyLocalizeStandardizeChar(p,obj.projMacros);
    end
    
    function projNewImStack(obj,ims,varargin)
      % DEVELOPMENT ONLY
      %
      % Same as projNew, but initialize project to have a single 'movie'
      % consisting of an image stack. 
      %
      % Optional PVs. Let N=numel(ims).
      % - xyGT. [Nxnptx2], gt labels. 
      % - xyTstT. [Nxnptx2xRT], CPR test results (replicats)
      % - xyTstTRed. [Nxnptx2], CPR test results (selected/final).
      % - tstITst. [K] indices into 1:N. If provided, xyTstT and xyTstTRed
      %   should have K rows. xyTstTITst specify frames to which tracking 
      %   results apply.
      %
      % If xyGT/xyTstT/xyTstTRed provided, they are viewed with
      % LabelCoreCPRView. 
      %
      % TODO: Prob should set pGT onto .labeledpos, and let
      % LabelCoreCPRView handle replicates etc.
      
      assert(false,'Unsupported');
      
      [xyGT,xyTstT,xyTstTRed,tstITst] = myparse(varargin,...
        'xyGT',[],...
        'xyTstT',[],...
        'xyTstTRed',[],...
        'tstITst',[]);

      assert(false,'Unsupported: todo gt');
      assert(~obj.isMultiView);
      
      obj.projNew('IMSTACK__DEVONLY');

      mr = MovieReaderImStack;
      mr.open(ims);
      obj.movieReader = mr;
      movieInfo = struct();
      movieInfo.nframes = mr.nframes;
      
      obj.movieFilesAll{end+1,1} = '__IMSTACK__';
      obj.movieFilesAllHaveLbls(end+1,1) = false; % note, this refers to .labeledpos
      obj.movieInfoAll{end+1,1} = movieInfo;
      obj.trxFilesAll{end+1,1} = '__IMSTACK__';
      obj.currMovie = 1; % HACK
      obj.currTarget = 1;
%       obj.labeledpos{end+1,1} = [];
%       obj.labeledpostag{end+1,1} = [];
      
      N = numel(ims);
      tfGT = ~isempty(xyGT);
      if tfGT
        [Ntmp,npt,d] = size(xyGT); % npt equal nLabelPoint?
        assert(Ntmp==N && d==2);
      end
      
      tfTst = ~isempty(xyTstT);
      tfITst = ~isempty(tstITst);
      if tfTst
        sz1 = size(xyTstT);
        sz2 = size(xyTstTRed);
        RT = size(xyTstT,4);
        if tfITst
          k = numel(tstITst);
          assert(isequal([k npt d],sz1(1:3),sz2));
          xyTstTPad = nan(N,npt,d,RT);
          xyTstTRedPad = nan(N,npt,d);
          xyTstTPad(tstITst,:,:,:) = xyTstT;
          xyTstTRedPad(tstITst,:,:) = xyTstTRed;
          
          xyTstT = xyTstTPad;
          xyTstTRed = xyTstTRedPad;
        else
          assert(isequal([N npt d],sz1(1:3),sz2));          
        end
      else
        xyTstT = nan(N,npt,d,1);
        xyTstTRed = nan(N,npt,d);
      end
      
      if tfGT
        lc = LabelCoreCPRView(obj);
        lc.setPs(xyGT,xyTstT,xyTstTRed);
        delete(obj.lblCore);
        obj.lblCore = lc;
        lpp = obj.labelPointsPlotInfo;
        lpp.Colors = obj.LabelPointColors();
        lc.init(obj.nLabelPoints,lpp);
        obj.setFrame(1);
      end
    end
            
  end
  
  methods (Static)
    
    function s = lblModernize(s)
      % s: struct, .lbl contents
      
	    % whether trackParams is stored -- update from 20190214
      isTrackParams = isfield(s,'trackParams');
      
      if ~isfield(s,'labeledposTS')
        nMov = numel(s.labeledpos);
        s.labeledposTS = cell(nMov,1);
        for iMov = 1:nMov
          lpos = s.labeledpos{iMov};
          [npts,~,nfrm,ntgt] = size(lpos);
          s.labeledposTS{iMov} = -inf(npts,nfrm,ntgt);
        end
        
        warningNoTrace('Label timestamps added (all set to -inf).');
      end
            
      if ~isfield(s,'labeledpos2')
        s.labeledpos2 = cellfun(@(x)nan(size(x)),s.labeledpos,'uni',0);
      end
      
      % 20160622
      if ~isfield(s,'nview') && ~isfield(s,'cfg')
        s.nview = 1;
      end
      if ~isfield(s,'viewNames') && ~isfield(s,'cfg')
        s.viewNames = {'1'};
      end

      % 20160629
      if isfield(s,'trackerClass')
        assert(isfield(s,'trackerData'));
      else
        if isfield(s,'CPRLabelTracker')
          s.trackerClass = 'CPRLabelTracker';
          s.trackerData = s.CPRLabelTracker;
        elseif isfield(s,'Interpolator')
          s.trackerClass = 'Interpolator';
          s.trackerData = s.Interpolator;
        else
          s.trackerClass = '';
          s.trackerData = [];
        end
      end
      
      % 20160707
      if ~isfield(s,'labeledposMarked')
        s.labeledposMarked = cellfun(@(x)false(size(x)),s.labeledposTS,'uni',0);
      end

      % 20160822 Modernize legacy projects that don't have a .cfg prop. 
      % Create a cfg from the lbl contents and fill in any missing fields 
      % with the current pref.yaml.
      if ~isfield(s,'cfg')
        % Create a config out what is in s. The large majority of config
        % info is not present in s; all other fields start from defaults.
        
        % first deal with multiview new def of NumLabelPoints
        nPointsReal = s.nLabelPoints/s.nview;
        assert(round(nPointsReal)==nPointsReal);
        s.nLabelPoints = nPointsReal;
        
        ptNames = arrayfun(@(x)sprintf('point%d',x),1:s.nLabelPoints,'uni',0);
        ptNames = ptNames(:);
        cfg = struct(...
          'NumViews',s.nview,...
          'ViewNames',{s.viewNames},...
          'NumLabelPoints',s.nLabelPoints,...
          'LabelPointNames',{ptNames},...
          'LabelMode',char(s.labelMode),...
          'LabelPointsPlot',s.labelPointsPlotInfo);
        fldsRm = {'nview' 'viewNames' 'nLabelPoints' 'labelMode' 'labelPointsPlotInfo'};
        s = rmfield(s,fldsRm);

        cfgbase = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
        if exist('pref.yaml','file')>0
          cfg1 = ReadYaml('pref.yaml');
          cfgbase = structoverlay(cfgbase,cfg1,'dontWarnUnrecog',true);
        end
        cfg = structoverlay(cfgbase,cfg,'dontWarnUnrecog',true);
        s.cfg = cfg;
      end
      s.cfg = Labeler.cfgModernize(s.cfg);
      
      % 20160816
      if isfield(s,'minv')
        assert(numel(s.minv)==numel(s.maxv));
        
        nminv = numel(s.minv);
        if nminv~=s.cfg.NumViews
          s.minv = repmat(s.minv(1),s.cfg.NumViews,1);
          s.maxv = repmat(s.maxv(1),s.cfg.NumViews,1);
        end
        
        % 20160927
        assert(isequal(numel(s.minv),numel(s.maxv),s.cfg.NumViews));
        for iView=1:s.cfg.NumViews
          s.cfg.View(iView).CLim.Min = s.minv(iView);
          s.cfg.View(iView).CLim.Max = s.maxv(iView);
        end
        s = rmfield(s,{'minv' 'maxv'});
      end
      
      % 20160927
      if isfield(s,'movieForceGrayscale')
        s.cfg.Movie.ForceGrayScale = s.movieForceGrayscale;
        s = rmfield(s,'movieForceGrayscale');
      end
      
      % 20161213
      if ~isfield(s,'viewCalProjWide')
        if ~isempty(s.viewCalibrationData)
          % Prior to today, all viewCalibrationDatas were always proj-wide
          s.viewCalProjWide = true;
          assert(isscalar(s.viewCalibrationData));
        else
          s.viewCalProjWide = [];
        end
      end
      
      % 20170808
      if ~isfield(s,'trackModeIdx')
        s.trackModeIdx = 1;
      end
      
      % 20170829
      GTPROPS = {
        'movieFilesAllGT'
        'movieInfoAllGT'
        'trxFilesAllGT'
        'labeledposGT'
        'labeledposTSGT'
        'labeledpostagGT'
        'viewCalibrationDataGT'
        'gtIsGTMode'
        'gtSuggMFTable'
        'gtTblRes'
      };
      tfGTProps = isfield(s,GTPROPS);
      allGTPresent = all(tfGTProps);
      noGTPresent = ~any(tfGTProps);
      assert(allGTPresent || noGTPresent);
      if noGTPresent
        nview = s.cfg.NumViews;
        s.movieFilesAllGT = cell(0,nview);
        s.movieInfoAllGT = cell(0,nview);
        s.trxFilesAllGT = cell(0,nview);
        s.labeledposGT = cell(0,1);
        s.labeledposTSGT = cell(0,1);
        s.labeledpostagGT = cell(0,1);
        if isscalar(s.viewCalProjWide) && s.viewCalProjWide
          s.viewCalibrationDataGT = [];
        else
          s.viewCalibrationDataGT = cell(0,1);
        end
        s.gtIsGTMode = false;
        s.gtSuggMFTable = MFTable.emptyTable(MFTable.FLDSID);
        s.gtTblRes = [];
      end

      % 20170922
      if ~isfield(s,'suspSelectedMFT')
        s.suspSelectedMFT = [];
      end
      if ~isfield(s,'suspComputeFcn')
        s.suspComputeFcn = [];
      end
      
      % 20171102
      if ~isfield(s,'xvResults')
        s.xvResults = [];
        s.xvResultsTS = [];
      end
      
      % 20171110
      for f={'labeledpostag' 'labeledpostagGT'},f=f{1}; %#ok<FXSET>
        val = s.(f);
        for i=1:numel(val)
          if iscell(val{i})
            val{i} = strcmp(val{i},'occ');
          end
        end
        s.(f) = val;
      end
      
      % 20180309 Preproc params
      % If preproc params are present in trackerData, move them to s and 
      % remove from trackerData
      % KB 20190214: only do this if trackParams not set
      if ~isTrackParams,

        % KB 20190214: leave preproc in here if this is after the 20190214 change to parameters
        tfTrackerDataHasPPParams = ~isempty(s.trackerData) && ...
          isstruct(s.trackerData) && ...
          isfield(s.trackerData,'sPrm') && ...
          ~isempty(s.trackerData.sPrm) && ...
          isfield(s.trackerData.sPrm,'PreProc');
        if isfield(s,'preProcParams')
          assert(isfield(s,'preProcH0'));
          assert(~tfTrackerDataHasPPParams);
        else
          if tfTrackerDataHasPPParams
            ppPrm = s.trackerData.sPrm.PreProc;
            s.trackerData.sPrm = rmfield(s.trackerData.sPrm,'PreProc');
            
            % 20180314 BackSub. Move backsub-related fields from NborMask to
            % BackSub subprop.
            if isfield(ppPrm,'NeighborMask')
              assert(~isfield(ppPrm,'BackSub'));
              ppPrm.BackSub.BGType = ppPrm.NeighborMask.BGType;
              ppPrm.BackSub.BGReadFcn = ppPrm.NeighborMask.BGReadFcn;
              ppPrm.NeighborMask = rmfield(ppPrm.NeighborMask,{'BGType' 'BGReadFcn'});
            end
          else
            ppPrm = struct();
          end
          
          s.preProcParams = ppPrm;
          s.preProcH0 = [];
        end
        
        ppPrm0 = APTParameters.defaultPreProcParamsOldStyle();
        if ~isempty(s.preProcParams)
          ppPrm1 = s.preProcParams;
        else
          ppPrm1 = struct();
        end
        [s.preProcParams,ppPrm0used] = structoverlay(ppPrm0,ppPrm1,...
          'dontWarnUnrecog',true);
        if ~isempty(ppPrm0used)
          fprintf('Using default preprocessing parameters for: %s.\n',...
            String.cellstr2CommaSepList(ppPrm0used));
        end        
      end
      
      % 20180525 DeepTrack integration. .trackerClass, .trackerData, .currTracker
      % 20181215 Updated for multiple DeepTrackers
      trkersInfo = LabelTracker.getAllTrackersCreateInfo;
      nDfltTrkers = numel(trkersInfo);
      if isempty(s.trackerClass)
        % Add current default trackers to all projs; doesn't hurt to have
        % them there
        s.trackerClass = trkersInfo;
        s.trackerData = repmat({[]},1,nDfltTrkers);
        s.currTracker = 0;
      elseif ischar(s.trackerClass)
        assert(strcmp(s.trackerClass,trkersInfo{1}{1}));
        assert(strcmp(s.trackerClass,'CPRLabelTracker'));
        s.trackerClass = trkersInfo;
        tData = repmat({[]},1,nDfltTrkers);
        tData{1} = s.trackerData;
        s.trackerData = tData;
        s.currTracker = 1;
      elseif iscell(s.trackerClass)
                
        nExistingTrkers = numel(s.trackerClass);
        if nExistingTrkers==2 % 20181214, only one deeptracker that had mutable trnNetType
          assert(isequal(s.trackerClass,{'CPRLabelTracker' 'DeepTracker'}));
          
          % KB 20181217 - this was stored as a char originally
          if ~isfield(s.trackerData{2},'trnNetType'),
            s.trackerData{2}.trnNetType = DLNetType.mdn;
          elseif ischar(s.trackerData{2}.trnNetType),
            s.trackerData{2}.trnNetType = DLNetType.(s.trackerData{2}.trnNetType);
          elseif isstruct(s.trackerData{2}.trnNetType), 
            %MK 20190110 - trnNetType can be a struct too.
            s.trackerData{2}.trnNetType = DLNetType.(s.trackerData{2}.trnNetType.ValueNames{1});  
          end
          
          dlTrkClsAug = {'DeepTracker' 'trnNetType' s.trackerData{2}.trnNetType};
          tf = cellfun(@(x)isequal(dlTrkClsAug,x),trkersInfo);
          iTrk = find(tf);
          assert(isscalar(iTrk));
          newTData = repmat({[]},1,nDfltTrkers);
          newTData{1} = s.trackerData{1};
          newTData{iTrk} = s.trackerData{2};
          s.trackerData = newTData;
          s.trackerClass = trkersInfo;
          if s.currTracker==2
            s.currTracker = iTrk;
          end
        else % 20181214, planning ahead when we add trackers
          [tf,loc] = LabelTracker.trackersCreateInfoIsMember(s.trackerClass(:),trkersInfo);
          assert(all(tf));
          tclass = trkersInfo;
          tclass(loc) = s.trackerClass(:);
          tdata = repmat({[]},1,nDfltTrkers);
          tdata(loc) = s.trackerData(:);
          s.trackerClass = tclass;
          s.trackerData = tdata;
          
          % KB 20201216 update currTracker as well
          oldCurrTracker = s.currTracker;
          if oldCurrTracker>0
            s.currTracker = loc(oldCurrTracker);
          end          
        end
      else
        assert(false);
      end
      
      % 20190207: added nLabels to dmc
      % 20190404: remove .trnName, .trnNameLbl as these dup DMC
      for i = 1:numel(s.trackerData),
        
%         AL20200312: This update will never work in the regular 
%         projLoad codepath because at this time the dmcs do not have 
%         their .rootDirs updated appropriately for the newly-exploded
%         bundled models. For now skip this update, .nLabels appears
%         noncritical (used for display/cosmetics only)
%         if isfield(s.trackerData{i},'trnLastDMC'),
%           for j = 1:numel(s.trackerData{i}.trnLastDMC),
%             dmc = s.trackerData{i}.trnLastDMC(j);
%             if isempty(dmc.nLabels) && (isempty(dmc.reader) || ~dmc.isRemote)
%               % dmc.reader is empty for legacy projs; which will be assumed 
%               % to be local in DeepTracker/modernizeSaveToken
%               try
%                 fprintf('Modernize: Reading nLabels for deep tracker\n');
%                 dmc.readNLabels();
%               catch ME
%                 warning('Could not read nLabels from trnLastDMC:\n%s',getReport(ME));
%               end
%             end
%           end
%         end

        if isfield(s.trackerData{i},'trnName') && ~isempty(s.trackerData{i}.trnName)
          if isfield(s.trackerData{i},'trnLastDMC') && ~isempty(s.trackerData{i}.trnLastDMC)
            assert(all(strcmp(s.trackerData{i}.trnName,...
                              {s.trackerData{i}.trnLastDMC.modelChainID})));
          end
          s.trackerData{i} = rmfield(s.trackerData{i},'trnName');
        end
        if isfield(s.trackerData{i},'trnNameLbl') && ~isempty(s.trackerData{i}.trnNameLbl)
          if isfield(s.trackerData{i},'trnLastDMC') && ~isempty(s.trackerData{i}.trnLastDMC)
            assert(all(strcmp(s.trackerData{i}.trnNameLbl,...
                              {s.trackerData{i}.trnLastDMC.trainID})));
          end
          s.trackerData{i} = rmfield(s.trackerData{i},'trnNameLbl');
        end
      end
      
      % 20180604
      if ~isfield(s,'labeledpos2GT')
        s.labeledpos2GT = cell(size(s.labeledposGT));
        for i=1:numel(s.labeledposGT)
          lposGTval = s.labeledposGT{i};
          if isstruct(lposGTval)
            s.labeledpos2GT{i} = SparseLabelArray.createEmpty(...
              lposGTval.size,lposGTval.type);
          else
            s.labeledpos2GT{i} = SparseLabelArray.createEmpty(...
              size(lposGTval),'nan');
          end
        end
      end
      
      % 20180619 Crop
      CROPFLDS = {'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' 'cropIsCropMode'};
      tfCropFlds = isfield(s,CROPFLDS);
      assert(all(tfCropFlds) || ~any(tfCropFlds));
      if ~any(tfCropFlds)
        s.cropIsCropMode = false;
        s.movieFilesAllCropInfo = cell(size(s.movieFilesAll,1),1);
        s.movieFilesAllGTCropInfo = cell(size(s.movieFilesAllGT,1),1);
      end
      
      % 20180706 movieReadPreLoadMovies
      if ~isfield(s,'movieReadPreLoadMovies')
        s.movieReadPreLoadMovies = false;
      end
      
      % 20180710 data cache
      if ~isfield(s,'preProcSaveData')
        s.preProcSaveData = false;
      end
      
      % 20180801 HistEqLUT
      LUTFLDS = {'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT'};
      tfLutFlds = isfield(s,LUTFLDS);
      assert(all(tfLutFlds) || ~any(tfLutFlds));
      if ~any(tfLutFlds)
        s.movieFilesAllHistEqLUT = cell(size(s.movieFilesAll));
        s.movieFilesAllGTHistEqLUT = cell(size(s.movieFilesAllGT));
      end
      
      % 20181022 projectHasTrx
      if ~isfield(s,'projectHasTrx'),
        s.projectHasTrx = ~isempty(s.trxFilesAll) && ~isempty(s.trxFilesAll{1});
      end
      
      % 20181101 movieInfo.readerobj (VideoReader) throwing warnings if
      % movs moved 
      for i=1:numel(s.movieInfoAll)
        if isfield(s.movieInfoAll{i}.info,'readerobj')
          % AL/SH XVid 20190328; Matlab apparent poor cleanup of
          % VideoReader objs
          rObj = s.movieInfoAll{i}.info.readerobj;
          if isobject(rObj)
            warningNoTrace('Deleting VideoReader obj with path %s/%s',...
              rObj.Path,rObj.Name);
            delete(rObj);
          end
          s.movieInfoAll{i}.info = rmfield(s.movieInfoAll{i}.info,'readerobj');
        end
      end
      
      % 20181215 factor dlbackend out of DeepTrackers into single/common
      % prop on Labeler
      if ~isfield(s,'trackDLBackEnd')
        % maybe change this by looking thru existing trackerDatas
        s.trackDLBackEnd = DLBackEndClass(DLBackEnd.Bsub);
      end
      % 20201028 docker/sing backend img/tag update
      s.trackDLBackEnd.modernize();
        
      
      % 20181220 DL common parameters
      if ~isTrackParams && ~isfield(s,'trackDLParams')
        cachedirs = cell(0,1);
        for i=2:numel(s.trackerData)
          td = s.trackerData{i};
          if ~isempty(td) && ~isempty(td.sPrm),
            if isfield(td.sPrm,'CacheDir'),
              cacheDir = td.sPrm.CacheDir;
            elseif isfield(td.sPrm,'Saving') && isfield(td.sPrm.Saving,'CacheDir'),
              cacheDir = td.sPrm.Saving.CacheDir;
            else
              cacheDir = '';
            end
            tfHasCache = ~isempty(cacheDir);
            if tfHasCache
              cachedirs{end+1,1} = cacheDir; %#ok<AGROW>
            end
          end
        end
        if ~isempty(cachedirs)
          if ~all(strcmp(cachedirs,cachedirs{1}))
            warningNoTrace('Project contains multiple DeepTracker cache directories: %s. Using first cache dir.',...          
             String.cellstr2CommaSepList(cachedirs));
          end
          cdir = cachedirs{1};
        else
          cdir = '';
        end
        % KB 20190212: set all common parameters, not just cachedir
        %s.trackDLParams = struct('CacheDir',cdir);
        s.trackDLParams = APTParameters.defaultParamsStructDTCommon;
        s.trackDLParams.Saving.CacheDir = cdir;
      end
      
      % 20190124 DL data cache; set
      % .preProcParams.TargetCrop.AlignUsingTrxTheta based on cpr parameter
      if s.preProcSaveData && ~isfield(s,'ppdb')
        s.ppdb = [];
      end
      
      if ~isTrackParams && ~isempty(s.trackerData{1})
        cprprms = s.trackerData{1}.sPrm;
        if ~isempty(cprprms) && isfield(cprprms.TrainInit,'usetrxorientation')
          % legacy project has 3-way enum param for cpr under .TrainInit and
          % .TestInit. Initialize .preProcParams...AlignUsingTrxTheta using
          % this val. Then remove these parameters now too although
          % CPRLT.modernizeParams would have done it.
          
          assert(~s.preProcParams.TargetCrop.AlignUsingTrxTheta); % default value added above
          s.preProcParams.TargetCrop.AlignUsingTrxTheta = cprprms.TrainInit.usetrxorientation;
          s.trackerData{1}.sPrm.TrainInit = rmfield(s.trackerData{1}.sPrm.TrainInit,'usetrxorientation');
          s.trackerData{1}.sPrm.TestInit = rmfield(s.trackerData{1}.sPrm.TestInit,'usetrxorientation');
          
          if s.preProcParams.TargetCrop.AlignUsingTrxTheta
            % .AlignUsingTrxTheta has mutated from default value. Any
            % existing DL cache and trackers need to be cleared
            s.ppdb = [];
            warningNoTrace('New preprocessing parameter .AlignUsingTrxTheta has been set to true. Clearing existing DL trackers; they will need to be retrained.');
            for iTrker=1:numel(s.trackerData)
              if strcmp(s.trackerClass{iTrker}{1},'DeepTracker') && ~isempty(s.trackerData{iTrker})
                s.trackerData{iTrker}.trnLastDMC = [];
                s.trackerData{iTrker}.movIdx2trkfile = containers.Map('keytype','int32','valuetype','any');
                warningNoTrace('Cleared Deep Learning tracker of type ''%s''.',char(s.trackerData{iTrker}.trnNetType));
              end
            end
          end
        end
      end
      
      % KB 20190212: reorganized DL parameters -- many specific parameters
      % were moved to common, and organized common parameters. leaf names
      % should all be the same, and unique, so just match leaves
      s = reorganizeDLParams(s); 
      
      % KB 20190214: all parameters are combined now
      if ~isTrackParams,
        s.trackParams = Labeler.trackGetParamsFromStruct(s);
      end
      
      % KB 20191218: replaced scale_range with scale_factor_range
      if isstruct(s.trackParams) && isfield(s.trackParams,'ROOT') && ...
          isstruct(s.trackParams.ROOT) && isfield(s.trackParams.ROOT,'DeepTrack') && ...
          isstruct(s.trackParams.ROOT.DeepTrack) && isfield(s.trackParams.ROOT.DeepTrack,'DataAugmentation') && ...
          isstruct(s.trackParams.ROOT.DeepTrack.DataAugmentation) && ...
          ~isfield(s.trackParams.ROOT.DeepTrack.DataAugmentation,'scale_factor_range') && ...
          isfield(s.trackParams.ROOT.DeepTrack.DataAugmentation,'scale_range'),
        if s.trackParams.ROOT.DeepTrack.DataAugmentation.scale_range ~= 0,
          warningNoTrace(['"Scale range" data augmentation parameter has been replaced by "Scale factor range". ' ...
            'These are very similar, so we have auto-populated "Scale factor range" based on your '...
            '"Scale range" parameter. However, these are not the same. Please examine "Scale factor range" '...
            'in the Tracking parameters GUI.']);
        end
        s.trackParams.ROOT.DeepTrack.DataAugmentation.scale_factor_range = 1+s.trackParams.ROOT.DeepTrack.DataAugmentation.scale_range;
      end
      
      % KB 20190331: adding in post-processing parameters if missing
      % AL 20190507: ... [a subset of] ... .trackParams modernization
      % AL 20190712: (subsumes above) modernizing entire .trackParams
      if ~isempty(s.trackParams)       
        sPrmDflt = APTParameters.defaultParamsStructAll;
        s.trackParams = structoverlay(sPrmDflt,s.trackParams,...
          'dontWarnUnrecog',true); % to allow removal of obsolete params
      else
        % s.trackParams can be [] for eg new projs, will be set at
        % parameter-set-time
      end
      
      % KB 20190214: store all parameters in each tracker so that we don't
      % have to delete trackers when tracking parameters change
      % AL 20190712: further clarification 
      for i = 1:numel(s.trackerData),
        if isempty(s.trackerData{i}),
          continue;
        end
        
%         tfCPRHasTrained = strcmp(s.trackerClass{i}{1},'CPRLabelTracker') ...
%                        && ~isempty(s.trackerData{i}.trnResRC) ...
%                        && any([s.trackerData{i}.trnResRC.hasTrained]);
%         tfDTHasTrained = strcmp(s.trackerClass{i}{1},'DeepTracker') ...
%                        && ~isempty(s.trackerData{i}.trnLastDMC);

        if ~isfield(s.trackerData{i},'sPrmAll') || isempty(s.trackerData{i}.sPrmAll),
          if isfield(s.trackerData{i},'sPrm') && ~isempty(s.trackerData{i}.sPrm)
            % legacy proj: .sPrm present
            
            tfCPR = strcmp(s.trackerClass{i}{1},'CPRLabelTracker');
            tfDT = strcmp(s.trackerClass{i}{1},'DeepTracker');
            s.trackerData{i}.sPrmAll = s.trackParams;
            
            if tfCPR
              CPRParams1 = APTParameters.all2CPRParams(s.trackerData{i}.sPrmAll,...
                numel(s.cfg.LabelPointNames),s.cfg.NumViews);
              assert(isequaln(CPRParams1,s.trackerData{i}.sPrm));
            elseif tfDT
              DLSpecificParams1 = APTParameters.all2DLSpecificParams(...
                s.trackerData{i}.sPrmAll,s.trackerClass{i}{3});
              assert(isequaln(DLSpecificParams1,s.trackerData{i}.sPrm));
            else
              assert(false);
            end
          else
            if ~isfield(s.trackerData{i},'sPrmAll')
              s.trackerData{i}.sPrmAll = [];
            end
          end
          
          if isfield(s.trackerData{i},'sPrm'),
            s.trackerData{i} = rmfield(s.trackerData{i},'sPrm');
          end          
        else
          % s.trackerData{i}.sPrmAll is present
          assert(~isfield(s.trackerData{i},'sPrm'),'Unexpected legacy parameters.');
        end
        
        % At this point for s.trackerData{i}:
        % 1. For legacy projs that had .sPrm but no .sPrmAll, we created 
        %    .sPrmAll and asserted that those params effectively match the 
        %    old .sPrm. Then we removed the .sPrm. In this case the 
        %   .sPrmAll happens to be modernized already.
        % 2. .sPrm has been removed in all cases.
        % 3. For modern projs that have .sPrmAll, this may not yet be
        % modernized. Responsibility for modernization will now be in the
        % LabelTrackers/loadSaveToken. Hmm not sure this is best. 
        
%         % KB 20190331: adding in post-processing parameters if missing
          % AL 20190712: This is now LabelTracker/loadSaveToken's
          % responsibility
%         if ~isempty(s.trackerData{i}.sPrmAll) && ...
%     	     ~isfield(s.trackerData{i}.sPrmAll.ROOT,'PostProcess'),
%           s.trackerData{i}.sPrmAll.ROOT.PostProcess = s.trackParams.ROOT.PostProcess;
%         end          
      end
      
      if isfield(s,'preProcParams'),
        s = rmfield(s,'preProcParams');
      end
      if isfield(s,'trackDLParams'),
        s = rmfield(s,'trackDLParams');
      end
            
      % KB 20190314: added skeleton
      if ~isfield(s,'skeletonEdges'),
        s.skeletonEdges = zeros(0,2);
      end
      if ~isfield(s,'showSkeleton'),
        s.showSkeleton = false;
      end

      % KB 20191203: added landmark matches
      if ~isfield(s,'flipLandmarkMatches'),
        s.flipLandmarkMatches = zeros(0,2);
      end

      
      % 20190429 TrkRes
      if ~isfield(s,'trkRes')
        s = Labeler.resetTrkResFieldsStruct(s);
      end
      if size(s.trkRes,1)~=size(s.movieFilesAll,1) || ...
         size(s.trkResGT,1)~=size(s.movieFilesAllGT,1) 
        % AL20200702: we appear to have had a bug in movieSetAdd which 
        % didn't maintain/update .trkRes/.trkResGT size appropriately. Fix
        % here at load-time. The .trkRes* infrastructure is power-users only
        % so probably very few regular users use it.
        warningNoTrace('Unexpected .trkRes* size. Resetting .trkRes* state. (This is not a commonly-used feature.)');
        s = Labeler.resetTrkResFieldsStruct(s);   
      end
    end
    function s = resetTrkResFieldsStruct(s)
      % see .trackResInit, maybe can combine
      nmov = size(s.movieFilesAll,1);
      nmovGT = size(s.movieFilesAllGT,1);
      nvw = size(s.movieFilesAll,2);
      s.trkResIDs = cell(0,1);
      s.trkRes = cell(nmov,nvw,0);
      s.trkResGT = cell(nmovGT,nvw,0);
      s.trkResViz = cell(0,1);
    end
    
    function data = stcLoadLblFile(fname)
      tname = tempname;
      try
        untar(fname,tname);
        data = load(fullfile(tname,'label_file.lbl'),'-mat');
        rmdir(tname,'s');
      catch ME,
        if strcmp(ME.identifier,'MATLAB:untar:invalidTarFile'),
          data = load(fname,'-mat');
        else
          throw(ME);
        end
      end
    end
        
  end 
  
  %% Movie
  methods
    
    function movieAdd(obj,moviefile,trxfile,varargin)
      % Add movie/trx to end of movie/trx list.
      %
      % moviefile: string or cellstr (can have macros)
      % trxfile: (optional) string or cellstr 
      
      notify(obj,'startAddMovie');      
      
      assert(~obj.isMultiView,'Unsupported for multiview labeling.');
      
      [offerMacroization,gt] = myparse(varargin,...
        'offerMacroization',~isdeployed&&obj.isgui, ... % If true, look for matches with existing macros
        'gt',obj.gtIsGTMode ... % If true, add moviefile/trxfile to GT lists. Could be a separate method, but there is a lot of shared code/logic.
        );
      
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      if exist('trxfile','var')==0 || isequal(trxfile,[])
        if ischar(moviefile)
          trxfile = '';
        elseif iscellstr(moviefile)
          trxfile = repmat({''},size(moviefile));
        else
          error('Labeler:movieAdd',...
            '''Moviefile'' must be a char or cellstr.');
        end
      end
      moviefile = cellstr(moviefile);
      trxfile = cellstr(trxfile);
      szassert(moviefile,size(trxfile));
      nMov = numel(moviefile);
        
      mr = MovieReader();
      mr.preload = obj.movieReadPreLoadMovies;
      for iMov = 1:nMov
        movFile = moviefile{iMov};
        tFile = trxfile{iMov};
        
        if offerMacroization 
          % Optionally replace movFile, tFile with macroized versions
          [tfCancel,macro,movFileMacroized] = ...
            FSPath.offerMacroization(obj.projMacros,{movFile});
          if tfCancel
            continue;
          end
          tfMacroize = ~isempty(macro);
          if tfMacroize
            assert(isscalar(movFileMacroized));
            movFile = movFileMacroized{1};
          end
          
          % trx
          % Note, tFile could already look like $movdir\trx.mat which would
          % be fine.
          movFileFull = obj.projLocalizePath(movFile);
          [tfMatch,tFileMacroized] = FSPath.tryTrxfileMacroization(...
            tFile,fileparts(movFileFull));
          if tfMatch
            tFile = tFileMacroized;
          end
        end
      
        movfilefull = obj.projLocalizePath(movFile);
        assert(exist(movfilefull,'file')>0,'Cannot find file ''%s''.',movfilefull);
        
        % See movieSetInProj()
        if any(strcmp(movFile,obj.(PROPS.MFA)))
          if nMov==1
            error('Labeler:dupmov',...
              'Movie ''%s'' is already in project.',movFile);
          else
            warningNoTrace('Labeler:dupmov',...
              'Movie ''%s'' is already in project and will not be added to project.',movFile);
            continue;
          end
        end
        if any(strcmp(movfilefull,obj.(PROPS.MFAF)))
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'', macro-expanded to ''%s'', is already in project.',...
            movFile,movfilefull);
        end
        
        tFileFull = Labeler.trxFilesLocalize(tFile,movfilefull);
        if ~(isempty(tFileFull) || exist(tFileFull,'file')>0)
          FSPath.throwErrFileNotFoundMacroAware(tFile,tFileFull,'trxfile');
        end

        % Could use movieMovieReaderOpen but we are just using MovieReader 
        % to get/save the movieinfo.
      
        mr.open(movfilefull); 
        ifo = struct();
        ifo.nframes = mr.nframes;
        ifo.info = mr.info;
        mr.close();
        
        if ~isempty(tFileFull)
          tmptrx = obj.getTrx(tFileFull,ifo.nframes);
          nTgt = numel(tmptrx);
        else
          nTgt = 1;
        end
        
        nlblpts = obj.nLabelPoints;
        nfrms = ifo.nframes;
        obj.(PROPS.MFA){end+1,1} = movFile;
        obj.(PROPS.MFAHL)(end+1,1) = 0;
        obj.(PROPS.MIA){end+1,1} = ifo;
        obj.(PROPS.MFACI){end+1,1} = CropInfo.empty(0,0);
        if obj.cropProjHasCrops
          wh = obj.cropGetCurrentCropWidthHeightOrDefault();
          obj.cropInitCropsGen(wh,PROPS.MIA,PROPS.MFACI,...
            'iMov',numel(obj.(PROPS.MFACI)));
        end
        obj.(PROPS.MFALUT){end+1,1} = [];
        obj.(PROPS.TFA){end+1,1} = tFile;
        obj.(PROPS.LPOS){end+1,1} = nan(nlblpts,2,nfrms,nTgt);
        obj.(PROPS.LPOSTS){end+1,1} = -inf(nlblpts,nfrms,nTgt);
        obj.(PROPS.LPOSTAG){end+1,1} = false(nlblpts,nfrms,nTgt);
        obj.(PROPS.LPOS2){end+1,1} = nan(nlblpts,2,nfrms,nTgt);
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          obj.(PROPS.VCD){end+1,1} = [];
        end
        obj.(PROPS.TRKRES)(end+1,:,:) = {[]};
        if ~gt
          obj.labeledposMarked{end+1,1} = false(nlblpts,nfrms,nTgt);
        end
      end
      
      notify(obj,'finishAddMovie');            
      
    end
    
    function movieAddBatchFile(obj,bfile)
      % Read movies from batch file
      
      if exist(bfile,'file')==0
        error('Labeler:movieAddBatchFile','Cannot find file ''%s''.',bfile);
      end
      movs = importdata(bfile);
      try
        movs = regexp(movs,',','split');
        movs = cat(1,movs{:});
      catch ME
        error('Labeler:batchfile',...
          'Error reading file %s: %s',bfile,ME.message);
      end
      if size(movs,2)~=obj.nview
        error('Labeler:batchfile',...
          'Expected file %s to have %d column(s), one for each view.',...
          bfile,obj.nview);
      end
      if ~iscellstr(movs)
        error('Labeler:movieAddBatchFile',...
          'Could not parse file ''%s'' for filenames.',bfile);
      end
      nMovSetImport = size(movs,1);
      if obj.nview==1
        fprintf('Importing %d movies from file ''%s''.\n',nMovSetImport,bfile);
        obj.movieAdd(movs,[],'offerMacroization',false);
      else
        fprintf('Importing %d movie sets from file ''%s''.\n',nMovSetImport,bfile);
        for i=1:nMovSetImport
          try
            obj.movieSetAdd(movs(i,:),'offerMacroization',false);
          catch ME
            warningNoTrace('Labeler:mov',...
              'Error trying to add movieset %d: %s Movieset not added to project.',...
              i,ME.message);
          end
        end
      end
    end

    function movieSetAdd(obj,moviefiles,varargin)
      % Add a set of movies (Multiview mode) to end of movie list.
      %
      % moviefiles: cellstr (can have macros)

      [offerMacroization,gt] = myparse(varargin,...
        'offerMacroization',true, ... % If true, look for matches with existing macros
        'gt',obj.gtIsGTMode ... % If true, add moviefiles to GT lists. 
        );

      if obj.nTargets~=1
        error('Labeler:movieSetAdd','Unsupported for nTargets>1.');
      end
      
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      moviefiles = cellstr(moviefiles);
      if numel(moviefiles)~=obj.nview
        error('Labeler:movieAdd',...
          'Number of moviefiles supplied (%d) must match number of views (%d).',...
          numel(moviefiles),obj.nview);
      end
      
      if offerMacroization
        [tfCancel,macro,moviefilesMacroized] = ...
          FSPath.offerMacroization(obj.projMacros,moviefiles);
        if tfCancel
          return;
        end
        tfMacroize = ~isempty(macro);
        if tfMacroize
          moviefiles = moviefilesMacroized;
        end
      end
      
      [tfmatch,imovmatch,tfMovsEq,moviefilesfull] = obj.movieSetInProj(moviefiles);
      if tfmatch
        error('Labeler:dupmov',...
          'Movieset matches current movieset %d in project.',imovmatch);
      end
      
      cellfun(@(x)assert(exist(x,'file')>0,'Cannot find file ''%s''.',x),moviefilesfull);      
      
      for iView=1:obj.nview
        iMFmatches = find(tfMovsEq(:,iView,1));
        iMFFmatches = find(tfMovsEq(:,iView,2));
        iMFFmatches = setdiff(iMFFmatches,iMFmatches);
        if ~isempty(iMFmatches)
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'' (view %d) already exists in project.',...
            moviefiles{iView},iView);
        end          
        if ~isempty(iMFFmatches)
          warningNoTrace('Labeler:dupmov',...
            'Movie ''%s'' (view %d), macro-expanded to ''%s'', is already in project.',...
            moviefiles{iView},iView,moviefilesfull{iView});
        end
      end
            
      ifos = cell(1,obj.nview);
      mr = MovieReader();
      mr.preload = obj.movieReadPreLoadMovies;
      for iView = 1:obj.nview
        % Could use movieMovieReaderOpen but we are just using MovieReader 
        % to get/save the movieinfo.
        mr.open(moviefilesfull{iView});
        ifo = struct();
        ifo.nframes = mr.nframes;
        ifo.info = mr.info;
        mr.close();
        ifos{iView} = ifo;
      end
      
      % number of frames must be the same in all movies
      nFrms = cellfun(@(x)x.nframes,ifos);
      if ~all(nFrms==nFrms(1))
        nframesstr = arrayfun(@num2str,nFrms,'uni',0);
        nframesstr = String.cellstr2CommaSepList(nframesstr);
        nFrms = min(nFrms);
        warningNoTrace('Labeler:movieSetAdd',...
          'Movies do not have the same number of frames: %s. The number of frames will be set to %d for this movieset.',...
          nframesstr,nFrms);
        for iView=1:obj.nview
          ifos{iView}.nframes = nFrms;
        end
      else
        nFrms = nFrms(1);
      end
      nTgt = 1;
      
      nLblPts = obj.nLabelPoints;
      obj.(PROPS.MFA)(end+1,:) = moviefiles(:)';
      obj.(PROPS.MFAHL)(end+1,1) = 0;
      obj.(PROPS.MIA)(end+1,:) = ifos;
      obj.(PROPS.MFACI){end+1,1} = CropInfo.empty(0,0);
      if obj.cropProjHasCrops
        wh = obj.cropGetCurrentCropWidthHeightOrDefault();
        obj.cropInitCropsGen(wh,PROPS.MIA,PROPS.MFACI,...
          'iMov',numel(obj.(PROPS.MFACI)));
      end
      obj.(PROPS.MFALUT)(end+1,:) = {[]};
      obj.(PROPS.TFA)(end+1,:) = repmat({''},1,obj.nview);
      obj.(PROPS.LPOS){end+1,1} = nan(nLblPts,2,nFrms,nTgt);
      obj.(PROPS.LPOSTS){end+1,1} = -inf(nLblPts,nFrms,nTgt);
      obj.(PROPS.LPOSTAG){end+1,1} = false(nLblPts,nFrms,nTgt);
      obj.(PROPS.LPOS2){end+1,1} = nan(nLblPts,2,nFrms,nTgt);
      if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
        obj.(PROPS.VCD){end+1,1} = [];
      end
      obj.(PROPS.TRKRES)(end+1,:,:) = {[]};
      if ~gt
        obj.labeledposMarked{end+1,1} = false(nLblPts,nFrms,nTgt);
      end
      
      % This clause does not occur in movieAdd(), b/c movieAdd is called
      % from UI functions which do this for the user. Currently movieSetAdd
      % does not have any UI so do it here.
      if ~obj.hasMovie && obj.nmoviesGTaware>0
        obj.movieSet(1,'isFirstMovie',true);
      end
    end

    function [tf,imovmatch,tfMovsEq,moviefilesfull] = movieSetInProj(obj,moviefiles)
      % Return true if a movie(set) already exists in the project
      %
      % moviefiles: either char if nview==1, or [nview] cellstr. Can
      %   contain macros.
      % 
      % tf: true if moviefiles exists in a row of .movieFilesAll or
      %   .movieFilesAllFull
      % imovsmatch: if tf==true, the matching movie index; otherwise
      %   indeterminate
      % tfMovsEq: [nmovset x nview x 2] logical. tfMovsEq{imov,ivw,j} is
      %   true if moviefiles{ivw} matches .movieFilesAll{imov,ivw} if j==1
      %   or .movieFilesAllFull{imov,ivw} if j==2. This output contains
      %   "partial match" info for multiview projs.
      %
      % This is GT aware and matches are searched for the current GT state.
      
      if ischar(moviefiles)
        moviefiles = {moviefiles};
      end
      
      nvw = obj.nview;
      assert(iscellstr(moviefiles) && numel(moviefiles)==nvw);
      
      moviefilesfull = cellfun(@obj.projLocalizePath,moviefiles,'uni',0);
      
      PROPS = obj.gtGetSharedProps();

      tfMFeq = arrayfun(@(x)strcmp(moviefiles{x},obj.(PROPS.MFA)(:,x)),1:nvw,'uni',0);
      tfMFFeq = arrayfun(@(x)strcmp(moviefilesfull{x},obj.(PROPS.MFAF)(:,x)),1:nvw,'uni',0);
      tfMFeq = cat(2,tfMFeq{:}); % [nmoviesetxnview], true when moviefiles matches movieFilesAll
      tfMFFeq = cat(2,tfMFFeq{:}); % [nmoviesetxnview], true when movfilefull matches movieFilesAllFull
      tfMovsEq = cat(3,tfMFeq,tfMFFeq); % [nmovset x nvw x 2]. 3rd dim is {mfa,mfaf}
      
      for j=1:2
        iAllViewsMatch = find(all(tfMovsEq(:,:,j),2));
        if ~isempty(iAllViewsMatch)
          assert(isscalar(iAllViewsMatch));
          tf = true;
          imovmatch = iAllViewsMatch;
          return;
        end
      end
      
      tf = false;
      imovmatch = [];
    end
    
%     function tfSucc = movieRmName(obj,movName)
%       % movName: compared to .movieFilesAll (macros UNreplaced)
%       assert(~obj.isMultiView,'Unsupported for multiview projects.');
%       iMov = find(strcmp(movName,obj.movieFilesAll));
%       if isscalar(iMov)
%         tfSucc = obj.movieRm(iMov);
%       end
%     end

    function tfSucc = movieRm(obj,iMov,varargin)
      % tfSucc: true if movie removed, false otherwise
      
      [force,gt] = myparse(varargin,...
        'force',false,... % if true, don't prompt even if mov has labels
        'gt',obj.gtIsGTMode ...
        );
      
      assert(isscalar(iMov));
      nMovOrig = obj.getnmoviesGTawareArg(gt);
      assert(any(iMov==1:nMovOrig),'Invalid movie index ''%d''.',iMov);
      if iMov==obj.currMovie
        error('Labeler:movieRm','Cannot remove current movie.');
      end
      
      tfProceedRm = true;
      haslbls1 = obj.labelPosMovieHasLabels(iMov,'gt',gt); % TODO: method should be unnec
      haslbls2 = obj.getMovieFilesAllHaveLblsArg(gt);
      haslbls2 = haslbls2(iMov);
      assert(haslbls1==haslbls2);
      if haslbls1 && ~obj.movieDontAskRmMovieWithLabels && ~force
        str = sprintf('Movie index %d has labels. Are you sure you want to remove?',iMov);
        BTN_NO = 'No, cancel';
        BTN_YES = 'Yes';
        BTN_YES_DAA = 'Yes, don''t ask again';
        btn = questdlg(str,'Movie has labels',BTN_NO,BTN_YES,BTN_YES_DAA,BTN_NO);
        if isempty(btn)
          btn = BTN_NO;
        end
        switch btn
          case BTN_NO
            tfProceedRm = false;
          case BTN_YES
            % none; proceed
          case BTN_YES_DAA
            obj.movieDontAskRmMovieWithLabels = true;
        end
      end
      
      if tfProceedRm
        PROPS = Labeler.gtGetSharedPropsStc(gt);
        nMovOrigReg = obj.nmovies;
        nMovOrigGT = obj.nmoviesGT;

        if gt
          movIdx = MovieIndex(-iMov);
          movIdxHasLbls = obj.movieFilesAllGTHaveLbls(iMov)>0;
        else
          movIdx = MovieIndex(iMov);
          movIdxHasLbls = obj.movieFilesAllHaveLbls(iMov)>0;
        end

        obj.(PROPS.MFA)(iMov,:) = [];
        obj.(PROPS.MFAHL)(iMov,:) = [];
        obj.(PROPS.MIA)(iMov,:) = [];
        obj.(PROPS.MFACI)(iMov,:) = [];
        obj.(PROPS.MFALUT)(iMov,:) = [];        
        obj.(PROPS.TFA)(iMov,:) = [];
        
        tfOrig = obj.isinit;
        obj.isinit = true; % AL20160808. we do not want set.labeledpos side effects, listeners etc.
        obj.(PROPS.LPOS)(iMov,:) = []; % should never throw with .isinit==true
        obj.(PROPS.LPOSTS)(iMov,:) = [];
        obj.(PROPS.LPOSTAG)(iMov,:) = [];
        obj.(PROPS.LPOS2)(iMov,:) = [];
        if ~gt
          obj.labeledposMarked(iMov,:) = [];
        end
        if isscalar(obj.viewCalProjWide) && ~obj.viewCalProjWide
          szassert(obj.(PROPS.VCD),[nMovOrig 1]);
          obj.(PROPS.VCD)(iMov,:) = [];
        end
        obj.(PROPS.TRKRES)(iMov,:,:) = [];

        obj.isinit = tfOrig;
        
        edata = MoviesRemappedEventData.movieRemovedEventData(...
          movIdx,nMovOrigReg,nMovOrigGT,movIdxHasLbls);
        obj.preProcData.movieRemap(edata.mIdxOrig2New);
        obj.ppdb.dat.movieRemap(edata.mIdxOrig2New);
        if gt
          [obj.gtSuggMFTable,tfRm] = MFTable.remapIntegerKey(...
            obj.gtSuggMFTable,'mov',edata.mIdxOrig2New);
          obj.gtSuggMFTableLbled(tfRm,:) = [];
          if ~isempty(obj.gtTblRes)
            obj.gtTblRes = MFTable.remapIntegerKey(obj.gtTblRes,'mov',...
              edata.mIdxOrig2New);
          end
          obj.notify('gtSuggUpdated');
          obj.notify('gtResUpdated');
        end
        
        obj.prevAxesMovieRemap(edata.mIdxOrig2New);
        
        % Note, obj.currMovie is not yet updated when this event goes out
        notify(obj,'movieRemoved',edata);
        
        if obj.currMovie>iMov && gt==obj.gtIsGTMode
          % AL 20200511. this may be overkill, maybe can just set 
          % .currMovie directly as the current movie itself cannot be 
          % rm-ed. A lot (if not all) state update here prob unnec
          obj.movieSet(obj.currMovie-1);
        end
      end
      
      tfSucc = tfProceedRm;
    end
    
    function movieRmAll(obj)
      nmov = obj.nmoviesGTaware;
      obj.movieSetNoMovie();
      for imov=1:nmov
        obj.movieRm(1,'force',true);
      end
    end
    
    function movieReorder(obj,p)
      % Reorder (regular/nonGT) movies 
      %
      % p: permutation of 1:obj.nmovies. Must contain all indices.
      
      nmov = obj.nmovies;
      p = p(:);
      if ~isequal(sort(p),(1:nmov)')
        error('Input argument ''p'' must be a permutation of 1..%d.',nmov);
      end
      
      tfSuspStuffEmpty = (isempty(obj.suspScore) || all(cellfun(@isempty,obj.suspScore))) ...
        && isempty(obj.suspSelectedMFT);
      if ~tfSuspStuffEmpty
        error('Reordering is currently unsupported for projects with suspiciousness.');
      end
      
      iMov0 = obj.currMovie;

      % Stage 1, outside obj.isinit block so listeners can update.
      % Future: clean up .isinit, listener policy etc it is getting too 
      % complex
      FLDS1 = {'movieInfoAll' 'movieFilesAll' 'movieFilesAllHaveLbls'...
        'movieFilesAllCropInfo' 'movieFileAllHistEqLUT' 'trxFilesAll'};
      for f=FLDS1,f=f{1}; %#ok<FXSET>
        obj.(f) = obj.(f)(p,:);
      end
      
      tfOrig = obj.isinit;
      obj.isinit = true;

      vcpw = obj.viewCalProjWide;
      if isempty(vcpw) || vcpw
        % none
      else
        obj.viewCalibrationData = obj.viewCalibrationData(p);
      end      
      FLDS2 = {...
        'labeledpos' 'labeledposTS' 'labeledposMarked' 'labeledpostag' ...
        'labeledpos2'};
      for f=FLDS2,f=f{1}; %#ok<FXSET>
        obj.(f) = obj.(f)(p,:);
      end
      obj.trkRes = obj.trkRes(p,:,:);
      
      obj.isinit = tfOrig;
      
      edata = MoviesRemappedEventData.moviesReorderedEventData(...
        p,nmov,obj.nmoviesGT);
      obj.preProcData.movieRemap(edata.mIdxOrig2New);
      obj.ppdb.dat.movieRemap(edata.mIdxOrig2New);
      notify(obj,'moviesReordered',edata);

      if ~obj.gtIsGTMode
        iMovNew = find(p==iMov0);
        obj.movieSet(iMovNew); %#ok<FNDSB>
      end
    end
    
    function movieFilesMacroize(obj,str,macro)
      % Replace a string with a macro throughout .movieFilesAll and *gt. A 
      % project macro is also added for macro->string.
      %
      % str: a string 
      % macro: macro which will replace all matches of string (macro should
      % NOT include leading $)
      
      if isfield(obj.projMacros,macro) 
        currVal = obj.projMacros.(macro);
        if ~strcmp(currVal,str)
          qstr = sprintf('Project macro ''%s'' is currently defined as ''%s''. This value can be redefined later if desired.',...
            macro,currVal);
          btn = questdlg(qstr,'Existing Macro definition','OK, Proceed','Cancel','Cancel');
          if isempty(btn)
            btn = 'Cancel';
          end
          switch btn
            case 'OK, Proceed'
              % none
            otherwise
              return;
          end           
        end
      end
        
      strpat = regexprep(str,'\\','\\\\');
      mfa0 = obj.movieFilesAll;
      mfagt0 = obj.movieFilesAllGT;
      if ispc
        mfa1 = regexprep(mfa0,strpat,['$' macro],'ignorecase');
        mfagt1 = regexprep(mfagt0,strpat,['$' macro],'ignorecase');
      else
        mfa1 = regexprep(mfa0,strpat,['$' macro]);
        mfagt1 = regexprep(mfagt0,strpat,['$' macro]);
      end
      obj.movieFilesAll = mfa1;
      obj.movieFilesAllGT = mfagt1;
      
      if ~isfield(obj.projMacros,macro) 
        obj.projMacroAdd(macro,str);
      end
    end
    
    function movieFilesUnMacroize(obj,macro)
      % "Undo" macro by replacing $<macro> with its value throughout
      % .movieFilesAll and *gt.
      %
      % macro: must be a currently defined macro
      
      if ~obj.projMacroIsMacro(macro)
        error('Labeler:macro','''%s'' is not a currently defined macro.',...
          macro);
      end
      
      sMacro = struct(macro,obj.projMacros.(macro));
      obj.movieFilesAll = FSPath.macroReplace(obj.movieFilesAll,sMacro);
      obj.movieFilesAllGT = FSPath.macroReplace(obj.movieFilesAllGT,sMacro);
    end
    
    function movieFilesUnMacroizeAll(obj)
      % Replace .movieFilesAll with .movieFilesAllFull and *gt; warn if a 
      % movie cannot be found
      
      mfaf = obj.movieFilesAllFull;
      mfagtf = obj.movieFilesAllGTFull;
      nmov = size(mfaf,1);
      nmovgt = size(mfagtf,1);
      nvw = obj.nview;
      for iView=1:nvw
        for iMov=1:nmov
          mov = mfaf{iMov,iView};
          if exist(mov,'file')==0
            warningNoTrace('Labeler:mov','Movie ''%s'' cannot be found.',mov);
          end
          obj.movieFilesAll{iMov,iView} = mov;
        end
        for iMov=1:nmovgt
          mov = mfagtf{iMov,iView};
          if exist(mov,'file')==0
            warningNoTrace('Labeler:mov','Movie ''%s'' cannot be found.',mov);
          end
          obj.movieFilesAllGT{iMov,iView} = mov;
        end
      end
      
      obj.projMacroClear;
    end
    
    function tfok = checkFrameAndTargetInBounds(obj,frm,tgt)
      tfok = false;
      if obj.nframes < frm,
        return;
      end
      if obj.hasTrx,
        if numel(obj.trx) < tgt,
          return;
        end
        trxtgt = obj.trx(tgt);
        if frm<trxtgt.firstframe || frm>trxtgt.endframe,
          return;
        end
      end
      
      tfok = true;
    end
    
    function [tfok,badfile] = movieCheckFilesExistSimple(obj,iMov,gt) % obj const
      % tfok: true if movie/trxfiles for iMov all exist, false otherwise.
      % badfile: if ~tfok, badfile contains a file that could not be found.

      PROPS = Labeler.gtGetSharedPropsStc(gt);
      
      if ~all(cellfun(@isempty,obj.(PROPS.TFA)(iMov,:)))
        assert(~obj.isMultiView,...
          'Multiview labeling with targets unsupported.');
      end
      
      for iView = 1:obj.nview
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        trxFileFull = obj.(PROPS.TFAF){iMov,iView};
        if exist(movfileFull,'file')==0
          tfok = false;
          badfile = movfileFull;
          return;
        elseif ~isempty(trxFileFull) && exist(trxFileFull,'file')==0
          tfok = false;
          badfile = trxFileFull;
          return;
        end
      end
      
      tfok = true;
      badfile = [];
    end
    
    function tfsuccess = movieCheckFilesExist(obj,iMov) % NOT obj const
      % Helper function for movieSet(), check that movie/trxfiles exist
      %
      % tfsuccess: false indicates user canceled or similar. True indicates
      % that i) obj.movieFilesAllFull(iMov,:) all exist; ii) if obj.hasTrx,
      % obj.trxFilesAllFull(iMov,:) all exist. User can update these fields
      % by browsing, updating macros etc.
      %
      % This function can harderror.
      %
      % This function is NOT obj const -- users can browse to 
      % movies/trxfiles, macro-related state can be mutated etc.
      
      
      tfsuccess = false;
      
      PROPS = obj.gtGetSharedProps();
      
      if ~all(cellfun(@isempty,obj.(PROPS.TFA)(iMov,:)))
        assert(~obj.isMultiView,...
          'Multiview labeling with targets unsupported.');
      end
                
      for iView = 1:obj.nview
        movfile = obj.(PROPS.MFA){iMov,iView};
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        
        if exist(movfileFull,'file')==0
          qstr = FSPath.errStrFileNotFoundMacroAware(movfile,...
            movfileFull,'movie');
          qtitle = 'Movie not found';
          if isdeployed || ~obj.isgui,
            error(qstr);
          end
          
          if FSPath.hasAnyMacro(movfile)
            qargs = {'Redefine macros','Browse to movie','Cancel','Cancel'};
          else
            qargs = {'Browse to movie','Cancel','Cancel'};
          end           
          resp = questdlg(qstr,qtitle,qargs{:});
          if isempty(resp)
            resp = 'Cancel';
          end
          switch resp
            case 'Cancel'
              return;
            case 'Redefine macros'
              obj.projMacroSetUI();
              movfileFull = obj.(PROPS.MFAF){iMov,iView};
              if exist(movfileFull,'file')==0
                emsg = FSPath.errStrFileNotFoundMacroAware(movfile,...
                  movfileFull,'movie');
                FSPath.errDlgFileNotFound(emsg);
                return;
              end
            case 'Browse to movie'
              pathguess = FSPath.maxExistingBasePath(movfileFull);
              if isempty(pathguess)
                pathguess = RC.getprop('lbl_lastmovie');
              end
              if isempty(pathguess)
                pathguess = pwd;
              end
              promptstr = sprintf('Select movie for %s',movfileFull);
              [newmovfile,newmovpath] = uigetfile('*.*',promptstr,pathguess);
              if isequal(newmovfile,0)
                return; % Cancel
              end
              movfileFull = fullfile(newmovpath,newmovfile);
              if exist(movfileFull,'file')==0
                emsg = FSPath.errStrFileNotFound(movfileFull,'movie');
                FSPath.errDlgFileNotFound(emsg);
                return;
              end
              
              % If possible, offer macroized movFile
              [tfCancel,macro,movfileMacroized] = ...
                FSPath.offerMacroization(obj.projMacros,{movfileFull});
              if tfCancel
                return;
              end
              tfMacroize = ~isempty(macro);
              if tfMacroize
                assert(isscalar(movfileMacroized));
                obj.(PROPS.MFA){iMov,iView} = movfileMacroized{1};
                movfileFull = obj.(PROPS.MFAF){iMov,iView};
              else
                obj.(PROPS.MFA){iMov,iView} = movfileFull;
              end
          end
          
          % At this point, either we have i) harderrored, ii)
          % early-returned with tfsuccess=false, or iii) movfileFull is set
          assert(exist(movfileFull,'file')>0);          
        end

        % trxfile
        %movfile = obj.(PROPS.MFA){iMov,iView};
        assert(strcmp(movfileFull,obj.(PROPS.MFAF){iMov,iView}));
        trxFile = obj.(PROPS.TFA){iMov,iView};
        trxFileFull = obj.(PROPS.TFAF){iMov,iView};
        tfTrx = ~isempty(trxFile);
        if tfTrx
          if exist(trxFileFull,'file')==0
            qstr = FSPath.errStrFileNotFoundMacroAware(trxFile,...
              trxFileFull,'trxfile');
            resp = questdlg(qstr,'Trxfile not found',...
              'Browse to trxfile','Cancel','Cancel');
            if isempty(resp)
              resp = 'Cancel';
            end
            switch resp
              case 'Browse to trxfile'
                % none
              case 'Cancel'
                return;
            end
            
            movfilepath = fileparts(movfileFull);
            promptstr = sprintf('Select trx file for %s',movfileFull);
            [newtrxfile,newtrxfilepath] = uigetfile('*.mat',promptstr,...
              movfilepath);
            if isequal(newtrxfile,0)
              return;
            end
            trxFile = fullfile(newtrxfilepath,newtrxfile);
            if exist(trxFile,'file')==0
              emsg = FSPath.errStrFileNotFound(trxFile,'trxfile');
              FSPath.errDlgFileNotFound(emsg);
              return;
            end
            [tfMatch,trxFileMacroized] = FSPath.tryTrxfileMacroization( ...
              trxFile,movfilepath);
            if tfMatch
              trxFile = trxFileMacroized;
            end
            obj.(PROPS.TFA){iMov,iView} = trxFile;
          end
          RC.saveprop('lbl_lasttrxfile',trxFile);
        end
      end
      
      % For multiview projs a user could theoretically alter macros in 
      % such a way as to incrementally locate files, breaking previously
      % found files
      for iView = 1:obj.nview
        movfile = obj.(PROPS.MFA){iMov,iView};
        movfileFull = obj.(PROPS.MFAF){iMov,iView};
        tfile = obj.(PROPS.TFA){iMov,iView};
        tfileFull = obj.(PROPS.TFAF){iMov,iView};
        if exist(movfileFull,'file')==0
          FSPath.throwErrFileNotFoundMacroAware(movfile,movfileFull,'movie');
        end
        if ~isempty(tfileFull) && exist(tfileFull,'file')==0
          FSPath.throwErrFileNotFoundMacroAware(tfile,tfileFull,'trxfile');
        end
      end
      
      tfsuccess = true;
    end
    
    function tfsuccess = movieSet(obj,iMov,varargin)
        
      notify(obj,'startSetMovie')
      % iMov: If multivew, movieSet index (row index into .movieFilesAll)
            
      assert(~isa(iMov,'MovieIndex')); % movieIndices, use movieSetMIdx
      assert(any(iMov==1:obj.nmoviesGTaware),...
                    'Invalid movie index ''%d''.',iMov);
      
      [isFirstMovie] = myparse(varargin,...
        'isFirstMovie',~obj.hasMovie... % passing true for the first time a movie is added to a proj helps the UI
        ); 
      
      tfsuccess = obj.movieCheckFilesExist(iMov); % throws
      if ~tfsuccess
        return;
      end
      
      movsAllFull = obj.movieFilesAllFullGTaware;
      cInfo = obj.movieFilesAllCropInfoGTaware{iMov};
      tfHasCrop = ~isempty(cInfo);

      ppPrms = obj.preProcParams;
      if ~isempty(ppPrms)
        bgsubPrms = ppPrms.BackSub;
        bgArgs = {'bgType',bgsubPrms.BGType,'bgReadFcn',bgsubPrms.BGReadFcn};
      else
        bgArgs = {};
      end        
      for iView=1:obj.nview
        mov = movsAllFull{iMov,iView};
        mr = obj.movieReader(iView);
        mr.preload = obj.movieReadPreLoadMovies;
        mr.open(mov,bgArgs{:}); % should already be faithful to .forceGrayscale, .movieInvert
        if tfHasCrop
          mr.setCropInfo(cInfo(iView)); % cInfo(iView) is a handle/pointer!
        else
          mr.setCropInfo([]);
        end
        RC.saveprop('lbl_lastmovie',mov);
        if iView==1
          if numel(obj.moviefile) > obj.MAX_MOVIENAME_LENGTH,
            obj.moviename = ['..',obj.moviefile(end-obj.MAX_MOVIENAME_LENGTH+3:end)];
          else
            obj.moviename = obj.moviefile;
          end
          %obj.moviename = FSPath.twoLevelFilename(obj.moviefile);
        end
      end
      
      % fix the clim so it doesn't keep flashing
      cmax_auto = nan(1,obj.nview); %#ok<*PROPLC>
      for iView = 1:obj.nview,
        im = obj.movieReader(iView).readframe(1,...
          'doBGsub',obj.movieViewBGsubbed,'docrop',false);
        cmax_auto(iView) = GuessImageMaxValue(im);
        if numel(obj.cmax_auto) >= iView && cmax_auto(iView) == obj.cmax_auto(iView) && ...
            size(obj.clim_manual,1) >= iView && all(~isnan(obj.clim_manual(iView,:))),
          set(obj.gdata.axes_all(iView),'CLim',obj.clim_manual(iView,:));
        else
          obj.clim_manual(iView,:) = nan;
          obj.cmax_auto(iView) = cmax_auto(iView);
          set(obj.gdata.axes_all(iView),'CLim',[0,cmax_auto(iView)]);
        end
      end
      
      isInitOrig = obj.isinit;
      obj.isinit = true; % Initialization hell, invariants momentarily broken
      obj.currMovie = iMov;
      
      if isFirstMovie,
        % KB 20161213: moved this up here so that we could redo in initHook
        obj.trkResVizInit();
        % we set template below as it requires .trx to be set correctly. 
        % see below
        obj.labelingInit('dosettemplate',false); 
      end
      
      % for fun debugging
      %       obj.gdata.axes_all.addlistener('XLimMode','PreSet',@(s,e)lclTrace('preset'));
      %       obj.gdata.axes_all.addlistener('XLimMode','PostSet',@(s,e)lclTrace('postset'));
      obj.setFrameAndTarget(1,1);
      
      trxFile = obj.trxFilesAllFullGTaware{iMov,1};
      tfTrx = ~isempty(trxFile);
      if tfTrx
        assert(~obj.isMultiView,...
          'Multiview labeling with targets is currently unsupported.');
        nfrm = obj.movieInfoAllGTaware{iMov,1}.nframes;
        trxvar = obj.getTrx(trxFile,nfrm);
      else
        trxvar = [];
      end
      obj.trxSet(trxvar);
      %obj.trxfile = trxFile; % this must come after .trxSet() call
        
      obj.isinit = isInitOrig; % end Initialization hell      

      % needs to be done after trx are set as labels2trkviz handles
      % multiple targets.
      % 20191017 done for every movie because different movies may have
      % diff numbers of targets.
      obj.labels2TrkVizInit();
      
      if isFirstMovie
        if obj.labelMode==LabelMode.TEMPLATE
          % Setting the template requires the .trx to be appropriately set,
          % so for template mode we redo this (it is part of labelingInit()
          % here.
          obj.labelingInitTemplate();
        end
      end

      % AL20160615: omg this is the plague.
      % AL20160605: These three calls semi-obsolete. new projects will not
      % have empty .labeledpos, .labeledpostag, or .labeledpos2 elements;
      % these are set at movieAdd() time.
      %
      % However, some older projects will have these empty els; and
      % maybe it's worth keeping the ability to have empty els for space
      % reasons (as opposed to eg filling in all els in lblModernize()).
      % Wait and see.
      % AL20170828 convert to asserts 
%       assert(~isempty(obj.labeledpos{iMov}));
%       assert(~isempty(obj.labeledposTS{iMov}));
%       assert(~isempty(obj.labeledposMarked{iMov}));
%       assert(~isempty(obj.labeledpostag{iMov}));
%       assert(~isempty(obj.labeledpos2{iMov}));
           
      edata = NewMovieEventData(isFirstMovie);
      notify(obj,'newMovie',edata);
      
      % Not a huge fan of this, maybe move to UI
      obj.updateFrameTableComplete();
      
      % Proj/Movie/LblCore initialization can maybe be improved
      % Call setFrame again now that lblCore is set up
      if obj.hasTrx
        obj.setFrameAndTarget(obj.currTrx.firstframe,obj.currTarget);
      else
        obj.setFrameAndTarget(1,1);
      end
            
    end
    
    function tfsuccess = movieSetMIdx(obj,mIdx,varargin)
      assert(isa(mIdx,'MovieIndex'));
      [iMov,gt] = mIdx.get();
      if gt~=obj.gtIsGTMode
        obj.gtSetGTMode(gt,'warnChange',true);
      end
      tfsuccess = obj.movieSet(iMov,varargin{:});
    end
    
    function movieSetNoMovie(obj,varargin)
      % Set .currMov to 0
                 
          % Stripped cut+paste form movieSet() for reference 20170714
          %       obj.movieReader(iView).open(movfileFull);
          %       obj.moviename = fullfile(parent,movname);
%       obj.isinit = true; % Initialization hell, invariants momentarily broken
          %       obj.currMovie = iMov;
          %       obj.setFrameAndTarget(1,1);
          %       obj.trxSet(trxvar);
          %       obj.trxfile = trxFile; % this must come after .trxSet() call
%       obj.isinit = false; % end Initialization hell
          %       obj.labelsMiscInit();
          %       obj.labelingInit();
          %       edata = NewMovieEventData(isFirstMovie);
          %       notify(obj,'newMovie',edata);
          %       obj.updateFrameTableComplete();
          %       if obj.hasTrx
          %         obj.setFrameAndTarget(obj.currTrx.firstframe,obj.currTarget);
          %       else
          %         obj.setFrameAndTarget(1,1);
          %       end
              
      for i=1:obj.nview
        obj.movieReader(i).close();
      end
      obj.moviename = '';
      %obj.trxfile = '';
      isInitOrig = obj.isinit;
      obj.isinit = true;
      obj.currMovie = 0;
      obj.trxSet([]);
      obj.currFrame = 1;
      obj.currTarget = 0;
      obj.isinit = isInitOrig;
      
      obj.labels2TrkVizInit();
      obj.trkResVizInit();
      obj.labelingInit('dosettemplate',false);
      edata = NewMovieEventData(false);
      notify(obj,'newMovie',edata);
      obj.updateFrameTableComplete();

      % Set state equivalent to obj.setFrameAndTarget();
      gd = obj.gdata;
      imsall = gd.images_all;
      for iView=1:obj.nview
        obj.currIm{iView} = 0;
        set(imsall(iView),'CData',0);
      end
      obj.prevIm = struct('CData',0,'XData',0,'YData',0);
      imprev = gd.image_prev;
      set(imprev,'CData',0);     
      obj.clearPrevAxesModeInfo();
      obj.currTarget = 1;
      obj.currFrame = 1;
      obj.prevFrame = 1;
      
%       obj.currSusp = [];
    end
    
    function s = moviePrettyStr(obj,mIdx)
      assert(isscalar(mIdx));
      [iMov,gt] = mIdx.get();
      if gt
        pfix = 'GT ';
      else
        pfix = '';
      end
      if obj.isMultiView
        mov = 'movieset';
      else
        mov = 'movie';
      end
      s = sprintf('%s%s %d',pfix,mov,iMov);
    end
    
    % Hist Eq
    
    function [hgram,hgraminfo] = movieEstimateImHist(obj,varargin) % obj CONST
      % Estimate the typical image histogram H0 of movies in the project.
      %
      % Operates on regular (non-GT) movies. Movies are sampled with all 
      % movies getting equal weight.
      %
      % If movies have crops, cropping occurs before histogram
      % counting/selection. Otoh Trx have no bearing on this method.
      %
      % hgram: [nbin] histogram count vector. See HistEq.selMovCentralImHist
      % hgraminfo: See HistEq.selMovCentralImHist
      
      [nFrmPerMov,nBinHist,iMovsSamp,debugViz] = myparse(varargin,...
        'nFrmPerMov',20,... % num frames to sample per mov
        'nBinHist',256, ... % num bins for imhist()
        'iMovsSamp',[],... % indices into .movieFilesAll to sample. defaults to 1:nmovies
        'debugViz',false ...
      );

      ppPrms = obj.preProcParams;
      if ~isempty(ppPrms) && ppPrms.BackSub.Use
        error('Unsupported when background subtraction is enabled.');
      end
    
      wbObj = WaitBarWithCancel('Histogram Equalization','cancelDisabled',true);
      oc = onCleanup(@()delete(wbObj));

      if isempty(iMovsSamp)
        iMovsSamp = 1:size(obj.movieFilesAll,1);
      end      
      nmovsetsSamp = numel(iMovsSamp);
      
      nvw = obj.nview;
      fread = nan(nFrmPerMov,nmovsetsSamp,nvw);
      cntmat = zeros(nBinHist,nmovsetsSamp,nvw);
      bins0 = [];
      for ivw=1:nvw
        wbstr = sprintf('Sampling movies, view %d',ivw);
        wbObj.startPeriod(wbstr,'shownumden',true,'denominator',nmovsetsSamp);
        mr = MovieReader;
        for i=1:nmovsetsSamp
          tic;
          wbObj.updateFracWithNumDen(i);
          
          imov = iMovsSamp(i);
          mIdx = MovieIndex(imov);
          obj.movieMovieReaderOpen(mr,mIdx,ivw);
          nfrmMov = mr.nframes;
          if nfrmMov<nFrmPerMov
            warningNoTrace('View %d, movie %d: sampling %d frames from a total of %d frames in movie.',...
              ivw,imov,nFrmPerMov,nfrmMov);
          end
          fsamp = linspace(1,nfrmMov,nFrmPerMov);
          fsamp = round(fsamp);
          fsamp = max(fsamp,1);
          fsamp = min(fsamp,nfrmMov);
          
          for iF=1:nFrmPerMov
            f = fsamp(iF);
            im = mr.readframe(f,'docrop',true);
            nchan = size(im,3);
            if nchan>1
              error('Images must be grayscale.');
            end

            [cnt,bins] = imhist(im,nBinHist);
            cntmat(:,i,ivw) = cntmat(:,i,ivw)+cnt;
            if isempty(bins0)
              bins0 = bins;
            elseif ~isequal(bins,bins0)
              dbins = unique(diff(bins));
              warningNoTrace('View %d, movie %d, frame %d: unexpected imhist bin vector. bin delta: %d',...
                ivw,imov,f,dbins);
            end
            fread(iF,i,ivw) = f;
          end
          
          t = toc;
          fprintf(1,'Elapsed time: %d sec\n',round(t));
        end
        wbObj.endPeriod();
      end

      [hgram,hgraminfo] = HistEq.selMovCentralImHist(cntmat,...
        'debugviz',debugViz);      
    end
    
    function movieEstimateHistEqLUTs(obj,varargin)
      % Update .movieFilesAllHistEqLUT, .movieFilesAllGTHistEqLUT based on
      % .preProcH0. Applying .movieFilesAllHistEqLUT{iMov} to frames of
      % iMov should have a hgram approximating .preProcH0
      
      [nFrmPerMov,wbObj,docheck] = myparse(varargin,...
        'nFrmPerMov',20, ... % num frames to sample per mov
        'wbObj',[],...
        'docheck',false...
        );
      
      if isempty(obj.preProcH0)
        error('No target image histogram set in property ''%s''.');
      end
            
      obj.movieEstimateHistEqLUTsHlp(false,nFrmPerMov,'docheck',docheck,'wbObj',wbObj);
      obj.movieEstimateHistEqLUTsHlp(true,nFrmPerMov,'docheck',docheck,'wbObj',wbObj);
    end
    function movieEstimateHistEqLUTsHlp(obj,isGT,nFrmPerMov,varargin)
      
      [wbObj,docheck] = myparse(varargin,...
        'wbObj',[],...
        'docheck',false);
      
      tfWB = ~isempty(wbObj);
      
      PROPS = obj.gtGetSharedPropsStc(isGT);
      nmovsets = obj.getnmoviesGTawareArg(isGT);
      nvw = obj.nview;

      obj.(PROPS.MFALUT) = cell(nmovsets,nvw);
      
      for ivw=1:nvw
        if tfWB
          wbstr = sprintf('Sampling movies, view %d',ivw);
          wbObj.startPeriod(wbstr,'shownumden',true,'denominator',nmovsets);
        end
        
        mr = MovieReader;
%         Isampcat = []; % used if debugViz
%         Jsampcat = []; % etc
%         Isampcatyoffs = 0;
        for imov=1:nmovsets
%           tic;
          if tfWB
            wbObj.updateFracWithNumDen(imov);
          end
          
          mIdx = MovieIndex(imov,isGT);
          obj.movieMovieReaderOpen(mr,mIdx,ivw);
          nfrmMov = mr.nframes;
          if nfrmMov<nFrmPerMov
            warningNoTrace('View %d, movie %d: sampling %d frames from a total of %d frames in movie.',...
              ivw,imov,nFrmPerMov,nfrmMov);
          end
          fsamp = linspace(1,nfrmMov,nFrmPerMov);
          fsamp = round(fsamp);
          fsamp = max(fsamp,1);
          fsamp = min(fsamp,nfrmMov);

          Isamp = cell(nFrmPerMov,1);
          for iF=1:nFrmPerMov
            f = fsamp(iF);
            im = mr.readframe(f,'docrop',true);
            nchan = size(im,3);
            if nchan>1
              error('Images must be grayscale.');
            end
            Isamp{iF} = im;
          end
          
          try
            Isamp = cat(2,Isamp{:});
          catch ME
            error('Cannot concatenate sampled movie frames: %s',ME.message);
          end
        
          hgram = obj.preProcH0.hgram(:,ivw);
          s = struct();
          s.fsamp = fsamp;
          s.hgram = hgram;
          [...
            s.lut,s.lutAL,...
            Ibin,s.binC,s.binE,s.intens2bin,...
            Jsamp,JsampAL,...
            Jbin,JbinAL,...
            s.hI,s.hJ,s.hJal,cI,cJ,cJal,...
            s.Tbin,s.TbinAL,Tbininv,TbininvAL] = ...
            HistEq.histMatch(Isamp,hgram,'docheck',docheck); %#ok<ASGLU>
          obj.(PROPS.MFALUT){imov,ivw} = s;
        
%           t = toc;
%           fprintf(1,'Elapsed time: %d sec\n',round(t));
        end
        
        if tfWB
          wbObj.endPeriod();
        end
      end
    end
    
    function movieHistEqLUTViz(obj)
      % Viz: 
      % - hgrams, cgrams, for each movie
      %   - hilite central hgram/cgram
      % - LUTs for all movs
      % - raw image montage Isamp
      % - sampled image montage
      %   - Jsamp
      %   - Jsamp2
      
      GT = false;
      %[iMovs,gt] = mIdx.get();
      mfaHEifos = obj.getMovieFilesAllHistEqLUTGTawareStc(GT);
      nmovs = obj.nmovies;
      nvw = obj.nview;
      
      if nmovs==0
        warningNoTrace('No movies specified.');
        return;
      end

      for ivw=1:nvw
        nbin = numel(mfaHEifos{1,ivw}.hgram);
        hI = nan(nbin,nmovs);
        hJ = nan(nbin,nmovs);
        hJal = nan(nbin,nmovs);
        Tbins = nan(nbin,nmovs);
        TbinALs = nan(nbin,nmovs);
        Isamp = [];
        Jsamp = [];
        JsampAL = [];
        Isampyoffs = 0;
        for imov=1:nmovs
          ifo = mfaHEifos{imov,ivw};
          hI(:,imov) = ifo.hI;
          hJ(:,imov) = ifo.hJ;
          hJal(:,imov) = ifo.hJal;
          Tbins(:,imov) = ifo.Tbin;
          TbinALs(:,imov) = ifo.TbinAL;
          
          mr = MovieReader;
          mIdx = MovieIndex(imov);
          obj.movieMovieReaderOpen(mr,mIdx,ivw);
          nfrms = numel(ifo.fsamp);
          Isampmov = cell(nfrms,1);
          Jsampmov = cell(nfrms,1);
          JsampALmov = cell(nfrms,1);
          for iF=1:nfrms
            f = ifo.fsamp(iF);
            im = mr.readframe(f,'docrop',true);
            nchan = size(im,3);
            if nchan>1
              error('Images must be grayscale.');
            end
            Isampmov{iF} = im;
            Jsampmov{iF} = ifo.lut(uint32(im)+1);
            JsampALmov{iF} = ifo.lutAL(uint32(im)+1);
          end
          Isampmov = cat(2,Isampmov{:});
          Jsampmov = cat(2,Jsampmov{:});
          JsampALmov = cat(2,JsampALmov{:});
          % normalize ims here before concating to account for possible
          % different classes
          Isampmov = HistEq.normalizeGrayscaleIm(Isampmov);
          Jsampmov = HistEq.normalizeGrayscaleIm(Jsampmov);
          JsampALmov = HistEq.normalizeGrayscaleIm(JsampALmov);
          
          Isamp = [Isamp; Isampmov]; %#ok<AGROW>
          Jsamp = [Jsamp; Jsampmov]; %#ok<AGROW>
          JsampAL = [JsampAL; JsampALmov]; %#ok<AGROW>
          Isampyoffs(end+1,1) = size(Isamp,1); %#ok<AGROW>
        end
        
        hgram = ifo.hgram;
        
        cgram = cumsum(hgram);
        cI = cumsum(hI);
        cJ = cumsum(hJ);
        cJal = cumsum(hJal);        
        
        x = 1:nbin;
        figure('Name','imhists and cdfs');

        axs = mycreatesubplots(2,3,.1);
        axes(axs(1,1));        
        plot(x,hI);
        hold on;
        grid on;
        hLines = plot(x,hgram,'linewidth',2);
%         legstr = sprintf('hI (%d movs)',nmovs);
        legend(hLines,{'hgram'});
        tstr = sprintf('Raw imhists (%d frms samp)',nfrms);
        title(tstr,'fontweight','bold');
        
        axes(axs(1,2));
        plot(x,hJ);
        hold on;
        grid on;
        hLines = plot(x,hgram,'linewidth',2);
%         legend(hLines,{'hgram'});
        tstr = sprintf('Xformed imhists');
        title(tstr,'fontweight','bold');
        
        axes(axs(1,3));
        plot(x,hJal);
        hold on;
        grid on;
        hLines = plot(x,hgram,'linewidth',2);
%         legend(hLines,{'hgram'});
        tstr = sprintf('Xformed (al) imhists');
        title(tstr,'fontweight','bold');

        axes(axs(2,1));        
        plot(x,cI);
        hold on;
        grid on;
        hLines = plot(x,cgram,'linewidth',2);
%         legstr = sprintf('cI (%d movs)',nmovs);
        legend(hLines,{'cgram'});
        tstr = sprintf('cdfs');
        title(tstr,'fontweight','bold');
        
        axes(axs(2,2));
        hLines = plot(x,cJ);
        hold on;
        grid on;
        hLines(end+1,1) = plot(x,cgram,'linewidth',2);
%         legend(hLines,{'cJ','hgram'});
        
        axes(axs(2,3));
        hLines = plot(x,cJal);
        hold on;
        grid on;
        hLines(end+1,1) = plot(x,cgram,'linewidth',2);
%         legend(hLines,{'cJal','cgram'});
        
        linkaxes(axs(1,:));
        linkaxes(axs(2,:));

        figure('Name','LUTs');
        x = (1:size(Tbins,1))';
        axs = mycreatesubplots(1,2,.1);
        axes(axs(1));
        plot(x,Tbins,'linewidth',2);
        grid on;
        title('Tbins','fontweight','bold');
        
        axes(axs(2));
        plot(x,TbinALs,'linewidth',2);
        grid on;
        title('TbinALs','fontweight','bold');
        
        linkaxes(axs);

        figure('Name','Sample Image Montage');
        axs = mycreatesubplots(3,1);
        axes(axs(1));
        imagesc(Isamp);
        colormap gray
        yticklocs = (Isampyoffs(1:end-1)+Isampyoffs(2:end))/2;
        yticklbls = arrayfun(@(x)sprintf('mov%d',x),1:nmovs,'uni',0);
        set(axs(1),'YTick',yticklocs,'YTickLabels',yticklbls);
        set(axs(1),'XTick',[]);
        tstr = sprintf('Raw images, view %d',ivw);
        if GT
          tstr = [tstr ' (gt)'];
        end
        title(tstr,'fontweight','bold');
        clim0 = axs(1).CLim;

        axes(axs(2));
        imagesc(Jsamp);
        colormap gray
%         colorbar
        axs(2).CLim = clim0;
        set(axs(2),'XTick',[],'YTick',[]);
        tstr = sprintf('Converted images, view %d',ivw);
        if GT
          tstr = [tstr ' (gt)'];
        end
        title(tstr,'fontweight','bold');
        
        axes(axs(3));
        imagesc(JsampAL);
        colormap gray
%         colorbar
        axs(3).CLim = clim0;
        set(axs(3),'XTick',[],'YTick',[]);
        tstr = sprintf('Converted images (AL), view %d',ivw);
        if GT
          tstr = [tstr ' (gt)'];
        end
        title(tstr,'fontweight','bold');
        
        linkaxes(axs);
      end
    end
    
    function J = movieHistEqApplyLUTs(obj,I,mIdxs,varargin)
      % Apply LUTs from .movieFilesAll*HistEqLUT to images
      %
      % I: [nmov x nview] cell array of raw grayscale images 
      % mIdxs: [nmov] MovieIndex vector labeling rows of I
      %
      % J: [nmov x nview] cell array of transformed/LUT-ed images
      
      wbObj = myparse(varargin,...
        'wbObj',[]);
      tfWB = ~isempty(wbObj);
      
      [nmov,nvw] = size(I);
      assert(nvw==obj.nview);
      assert(isa(mIdxs,'MovieIndex'));
      assert(isvector(mIdxs) && numel(mIdxs)==nmov);
      
      J = cell(size(I));      
      mIdxsUn = unique(mIdxs);
      for mi = mIdxsUn(:)'
        mfaHEifo = obj.getMovieFilesAllHistEqLUTMovIdx(mi);
        assert(isrow(mfaHEifo));
        rowsThisMov = find(mIdxs==mi);
        for ivw=1:nvw
          lut = mfaHEifo{ivw}.lutAL;
          lutcls = class(lut);
          for row=rowsThisMov(:)'
            im = I{row,ivw};
            assert(isa(im,lutcls));
            J{row,ivw} = lut(uint32(im)+1);
          end
        end
      end
    end
      
%     function movieHistEqLUTEffectMontageHlp(obj,isGT,frm,wbObj)
%       % OBSOLETE
%       PROPS = obj.gtGetSharedPropsStc(isGT);
%       Tbins = obj.(PROPS.MFALUT);
%       assert(~isempty(Tbins));
%       nmovsets = obj.getnmoviesGTawareArg(isGT);
%       nvw = obj.nview;
%       for ivw=1:nvw
%         wbstr = sprintf('Sampling movies, view %d',ivw);
%         wbObj.startPeriod(wbstr,'shownumden',true,'denominator',nmovsets);
%         mr = MovieReader;
%         Isamp = cell(nmovsets,1);
%         Jsamp = cell(nmovsets,1);
%         for imov=1:nmovsets
%           wbObj.updateFracWithNumDen(imov);          
%           mIdx = MovieIndex(imov,isGT);
%           obj.movieMovieReaderOpen(mr,mIdx,ivw);
%           Isamp{imov} = mr.readframe(frm,'docrop',true);
%           if size(Isamp{imov},3)>1
%             error('Image must be grayscale.');
%           end
%           lut = Tbins{imov,ivw};
%           Jsamp{imov} = lut(uint32(Isamp{imov})+1);
%         end
%         wbObj.endPeriod();
%         
%         figure;
%         ax = axes;
%         montage(cat(4,Isamp{:}),'Parent',ax);
% %         colormap gray
% %         colorbar
% %         yticklocs = (Isampcatyoffs(1:end-1)+Isampcatyoffs(2:end))/2;
% %         yticklbls = arrayfun(@(x)sprintf('mov%d',x),1:nmovsets,'uni',0);
% %         set(ax,'YTick',yticklocs,'YTickLabels',yticklbls);
%         tstr = sprintf('Raw images, view %d',ivw);
%         if isGT
%           tstr = [tstr ' (gt)']; %#ok<AGROW>
%         end
%         title(tstr,'fontweight','bold');
%         clim0 = ax.CLim;
%         
%         figure;
%         ax = axes;
%         montage(cat(4,Jsamp{:}),'Parent',ax);
% %         colormap gray
% %         colorbar
% %         set(ax,'YTick',yticklocs,'YTickLabels',yticklbls);
%         tstr = sprintf('Corrected images, view %d',ivw);
%         if isGT
%           tstr = [tstr ' (gt)']; %#ok<AGROW>
%         end
%         title(tstr,'fontweight','bold');
%         ax.CLim = clim0;
%       end
%     end    
    
    
    function movieMovieReaderOpen(obj,movRdr,mIdx,iView) % obj CONST
      % Take a movieReader object and open the movie (mIdx,iView), being 
      % faithful to obj as per:
      %   - .movieForceGrayScale 
      %   - .movieInvert(iView)
      %   - .preProcParams.BackSub
      %   - .cropInfo for (mIdx,iView) as appropriate
      %
      % movRdr: scalar MovieReader object
      % mIdx: scalar MovieIndex
      % iView: view index; used for .movieInvert

      ppPrms = obj.preProcParams;
      if ~isempty(ppPrms)
        bgsubPrms = ppPrms.BackSub;
        bgArgs = {'bgType',bgsubPrms.BGType,'bgReadFcn',bgsubPrms.BGReadFcn};
      else
        bgArgs = {};
      end
      
      movfname = obj.getMovieFilesAllFullMovIdx(mIdx);
      movRdr.preload = obj.movieReadPreLoadMovies; % must occur before .open()
      movRdr.open(movfname{iView},bgArgs{:});
      movRdr.forceGrayscale = obj.movieForceGrayscale;
      movRdr.flipVert = obj.movieInvert(iView);      
      cInfo = obj.getMovieFilesAllCropInfoMovIdx(mIdx);
      if ~isempty(cInfo)
        movRdr.setCropInfo(cInfo(iView));
      else
        movRdr.setCropInfo([]);
      end      
    end
    
    function movieSetMovieReadPreLoadMovies(obj,tf)
      tf0 = obj.movieReadPreLoadMovies;
      if tf0~=tf && (obj.nmovies>0 || obj.nmoviesGT>0)        
        warningNoTrace('Project already has movies. Checking movie lengths under preloading.');
        obj.hlpCheckWarnMovieNFrames('movieFilesAll','movieInfoAll',true,'');
        obj.hlpCheckWarnMovieNFrames('movieFilesAllGT','movieInfoAllGT',true,' (gt)');
      end
      obj.movieReadPreLoadMovies = tf;
    end
    function hlpCheckWarnMovieNFrames(obj,movFileFld,movIfoFld,preload,gtstr)
      mr = MovieReader;
      mr.preload = preload;
      movfiles = obj.(movFileFld);
      movifo = obj.(movIfoFld);
      [nmov,nvw] = size(movfiles);
      for imov=1:nmov
        fprintf('Movie %d%s\n',imov,gtstr);
        for ivw=1:nvw
          mr.open(movfiles{imov,ivw});
          nf = mr.nframes;
          nf0 = movifo{imov,ivw}.nframes;
          if nf~=nf0
            warningNoTrace('Movie %d%s view %d, old nframes=%d, new nframes=%d.',...
              imov,gtstr,ivw,nf0,nf);
          end
        end
      end
    end
    
  end
  
  %% Trx
  methods

    % TrxCache notes
    % To avoid repeated loading of trx data from filesystem, we cache
    % previously seen/loaded trx data in .trxCache. The cache is never
    % updated as it is assumed that trxfiles on disk do not mutate over the
    % course of a single APT session.
    
    function [trx,frm2trx] = getTrx(obj,filename,varargin)
      % [trx,frm2trx] = getTrx(obj,filename,[nfrm])
      % Get trx data for iMov/iView from .trxCache; load from filesys if
      % necessary      
      [trx,frm2trx] = Labeler.getTrxCacheStc(obj.trxCache,filename,varargin{:});
    end
    
    function clearTrxCache(obj)
      % Forcibly clear .trxCache
      obj.trxCache = containers.Map();
    end
  end
  methods (Static)
    function [trx,frm2trx] = getTrxCacheStc(trxCache,filename,nfrm)
      % Get trx data for iMov/iView from .trxCache; load from filesys if
      % necessary
      %
      % trxCache: containers.Map
      % filename: fullpath to trxfile
      % nfrm: total number of frames in associated movie
      %
      % trx: struct array
      % frm2trx: [nfrm x ntrx] logical. frm2trx(f,i) is true if trx(i) is
      %  live @ frame f
      
      if nargin < 3,
        nfrm = [];
      end
      
      if trxCache.isKey(filename)
        s = trxCache(filename);
        trx = s.trx;
        frm2trx = s.frm2trx;
        if isempty(nfrm),
          nfrm = size(frm2trx,1);
        end
        szassert(frm2trx,[nfrm numel(trx)]);
          
      else
        if exist(filename,'file')==0
          % Currently user will have to navigate to iMov to fix
          error('Labeler:file','Cannot find trxfile ''%s''.',filename);
        end
        tmp = load(filename,'-mat','trx');
        if isfield(tmp,'trx')
          trx = tmp.trx;
          frm2trx = Labeler.trxHlpComputeF2t(nfrm,trx);          
          trxCache(filename) = struct('trx',trx,'frm2trx',frm2trx); %#ok<NASGU>
          RC.saveprop('lbl_lasttrxfile',filename);
        else
          warningNoTrace('Labeler:trx',...
            'No ''trx'' variable found in trxfile %s.',filename);
          trx = [];
          frm2trx = [];
        end
      end
    end
    
    function [trxCell,frm2trxCell,frm2trxTotAnd] = ...
                          getTrxCacheAcrossViewsStc(trxCache,filenames,nfrm)
      % Similar to getTrxCacheStc, but for an array of filenames and with
      % some checks
      %
      % filenames: cellstr of trxfiles containing trx with the same total 
      %   frame number, eg trx across all views in a single movieset
      % nfrm: common number of frames
      % 
      % trxCell: cell array, same size as filenames, containing trx
      %   structarrays
      % frm2trxCell: cell array, same size as filenames, containing
      %   frm2trx arrays for each trx
      % frm2trxTotAnd: AND(frm2trxCell{:})
      
      assert(~isempty(filenames));
      
      [trxCell,frm2trxCell] = cellfun(@(x)Labeler.getTrxCacheStc(trxCache,x,nfrm),...
          filenames,'uni',0);
      cellfun(@(x)assert(numel(x)==numel(trxCell{1})),trxCell);

      % In multiview multitarget projs, the each view's trx must contain
      % the same number of els and these elements must correspond across
      % views.
      nTrx = numel(filenames);
      if nTrx>1 && isfield(trxCell{1},'id')
        trxids = cellfun(@(x)[x.id],trxCell,'uni',0);
        assert(isequal(trxids{:}),'Trx ids differ.');
      end
      
      frm2trxTotAnd = cellaccumulate(frm2trxCell,@and);
    end
  end
  methods
        
    function trxSet(obj,trx)
      % Set the trajectories for the current movie (.trx).
      %
      % - Call trxSave if you want the new trx to persist somewhere. 
      % Otherwise it will be gone once you switch movies.
      % - This leaves .trxfile and .trxFilesAll unchanged. 
      % - You probably want to call setFrameAndTarget() after this.
      % - This method CANNOT CHANGE the number of trx-- that would require
      % updating .labeledpos and .suspScore as well.
      % - (TODO) Warnings will be thrown if the various .firstframe/.endframes
      % change in such a way that existing labels now fall outside their
      % trx's existence.
      
      if ~obj.isinit
        % AL: seems bizzare, would you ever swap out the trx for the
        % current movie?
        assert(isequal(numel(trx),obj.nTargets)); 

        % TODO: check for labels that are "out of bounds" for new trx
      end
      
      obj.trx = trx;

      if obj.hasTrx
        if ~isfield(obj.trx,'id')
          for i = 1:numel(obj.trx)
            obj.trx(i).id = i;
          end
        end
        maxID = max([obj.trx.id]);
      else
        maxID = -1;
      end
%       id2t = nan(maxID+1,1);
%       for i = 1:obj.nTrx
%         id2t(obj.trx(i).id+1) = i;
%       end
%       obj.trxIdPlusPlus2Idx = id2t;
      if isnan(obj.nframes)
        obj.frm2trx = [];
      else
        % Can get this from trxCache
        obj.frm2trx = Labeler.trxHlpComputeF2t(obj.nframes,trx);
      end
      
      obj.currImHud.updateReadoutFields('hasTgt',obj.hasTrx);
      obj.initShowTrx();
    end
       
    function tf = trxCheckFramesLive(obj,frms)
      % Check that current target is live for given frames
      %
      % frms: [n] vector of frame indices
      %
      % tf: [n] logical vector
      
      iTgt = obj.currTarget;
      if obj.hasTrx
        tf = obj.frm2trx(frms,iTgt);
      else
        tf = true(numel(frms),1);
      end
    end
    
    function trxCheckFramesLiveErr(obj,frms)
      % Check that current target is live for given frames; err if not
      
      tf = obj.trxCheckFramesLive(frms);
      if ~all(tf)
        error('Labeler:target',...
          'Target %d is not live during specified frames.',obj.currTarget);
      end
    end   
    
    function trxFilesUnMacroize(obj)
      obj.trxFilesAll = obj.trxFilesAllFull;
      obj.trxFilesAllGT = obj.trxFilesAllGTFull;
    end
    
    function [tfok,tblBig] = hlpTargetsTableUIgetBigTable(obj)
      wbObj = WaitBarWithCancel('Target Summary Table');
      centerOnParentFigure(wbObj.hWB,obj.hFig);
      oc = onCleanup(@()delete(wbObj));      
      tblBig = obj.trackGetBigLabeledTrackedTable('wbObj',wbObj);
      tfok = ~wbObj.isCancel;
      % if ~tfok, tblBig indeterminate
    end
    function hlpTargetsTableUIupdate(obj,navTbl)
      [tfok,tblBig] = obj.hlpTargetsTableUIgetBigTable();
      if tfok
        navTbl.setData(obj.trackGetSummaryTable(tblBig));
      end
    end
    function targetsTableUI(obj)
      [tfok,tblBig] = obj.hlpTargetsTableUIgetBigTable();
      if ~tfok
        return;
      end
      
      tblSumm = obj.trackGetSummaryTable(tblBig);
      hF = figure('Name','Target Summary (click row to navigate)',...
        'MenuBar','none','Visible','off');
      hF.Position(3:4) = [1280 500];
      centerfig(hF,obj.hFig);
      hPnl = uipanel('Parent',hF,'Position',[0 .08 1 .92],'Tag','uipanel_TargetsTable');
      BTNWIDTH = 100;
      DXY = 4;
      btnHeight = hPnl.Position(2)*hF.Position(4)-2*DXY;
      btnPos = [hF.Position(3)-BTNWIDTH-DXY DXY BTNWIDTH btnHeight];      
      hBtn = uicontrol('Style','pushbutton','Parent',hF,...
        'Position',btnPos,'String','Update',...
        'fontsize',12);
      FLDINFO = {
        'mov' 'Movie' 'integer' 30
        'iTgt' 'Target' 'integer' 30
        'trajlen' 'Traj. Length' 'integer' 45
        'frm1' 'Start Frm' 'integer' 30
        'nFrmLbl' '# Frms Lbled' 'integer' 60
        'nFrmTrk' '# Frms Trked' 'integer' 60
        'nFrmImported' '# Frms Imported' 'integer' 90
        'nFrmLblTrk' '# Frms Lbled&Trked' 'integer' 120
        'lblTrkMeanErr' 'Track Err' 'float' 60
        'nFrmLblImported' '# Frms Lbled&Imported' 'integer'  120
        'lblImportedMeanErr' 'Imported Err' 'float' 60
        'nFrmXV' '# Frms XV' 'integer' 40
        'xvMeanErr' 'XV Err' 'float' 40};
      tblfldsassert(tblSumm,FLDINFO(:,1));
      nt = NavigationTable(hPnl,[0 0 1 1],...
        @(row,rowdata)obj.setMFT(rowdata.mov,rowdata.frm1,rowdata.iTgt),...
        'ColumnName',FLDINFO(:,2)',...
        'ColumnFormat',FLDINFO(:,3)',...
        'ColumnPreferredWidth',cell2mat(FLDINFO(:,4)'));
%      jt = nt.jtable;
      nt.setData(tblSumm);
%      cr.setHorizontalAlignment(javax.swing.JLabel.CENTER);
%      h = jt.JTable.getTableHeader;
%      h.setPreferredSize(java.awt.Dimension(225,22));
%      jt.JTable.repaint;
      
      hF.UserData = nt;
      hBtn.Callback = @(s,e)obj.hlpTargetsTableUIupdate(nt);
      hF.Units = 'normalized';
      hBtn.Units = 'normalized';
      hF.Visible = 'on';

      obj.addDepHandle(hF);
    end
        
  end
  methods (Static)
    function f2t = trxHlpComputeF2t(nfrm,trx)
      % Compute f2t array for trx structure
      %
      % nfrm: number of frames in movie corresponding to trx
      % trx: trx struct array
      %
      % f2t: [nfrm x nTrx] logical. f2t(f,iTgt) is true iff trx(iTgt) is
      % live at frame f.
      
      if isempty(nfrm),
        nfrm = max([trx.endframe]);
      end
      nTrx = numel(trx);
      f2t = false(nfrm,nTrx);
      for iTgt=1:nTrx
        frm0 = trx(iTgt).firstframe;
        frm1 = trx(iTgt).endframe;
        f2t(frm0:frm1,iTgt) = true;
      end
    end
    function sMacro = trxFilesMacros(movFileFull)
      sMacro = struct();
      sMacro.movdir = fileparts(movFileFull);
    end
    function trxFilesFull = trxFilesLocalize(trxFiles,movFilesFull)
      % Localize trxFiles based on macros+movFiles
      %
      % trxFiles: can be char or cellstr
      % movFilesFull: like trxFiles, if cellstr same size as trxFiles.
      %   Should NOT CONTAIN macros
      %
      % trxFilesFull: char or cellstr, if cellstr same size as trxFiles
      
      tfChar = ischar(trxFiles);
      trxFiles = cellstr(trxFiles);
      movFilesFull = cellstr(movFilesFull);
      szassert(trxFiles,size(movFilesFull));
      trxFilesFull = cell(size(trxFiles));
      
      for i=1:numel(trxFiles)
        sMacro = Labeler.trxFilesMacros(movFilesFull{i});
        trxFilesFull{i} = FSPath.fullyLocalizeStandardizeChar(trxFiles{i},sMacro);
      end
      FSPath.warnUnreplacedMacros(trxFilesFull);
      
      if tfChar
        trxFilesFull = trxFilesFull{1};
      end
    end
  end  
  methods % show*
    
    function initShowTrx(obj)
      deleteValidHandles(obj.hTraj);
      deleteValidHandles(obj.hTrx);
%       deleteValidHandles(obj.hTrxEll);
      deleteValidHandles(obj.hTrxTxt);
      obj.hTraj = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrx = matlab.graphics.primitive.Line.empty(0,1);
%       obj.hTrxEll = matlab.graphics.primitive.Line.empty(0,1);
      obj.hTrxTxt = matlab.graphics.primitive.Text.empty(0,1);
      
      ax = obj.gdata.axes_curr;
      pref = obj.projPrefs.Trx;
      for i = 1:obj.nTrx
        
        obj.hTraj(i,1) = line(...
          'parent',ax,...
          'xdata',nan, ...
          'ydata',nan, ...
          'color',pref.TrajColor,...
          'linestyle',pref.TrajLineStyle, ...
          'linewidth',pref.TrajLineWidth, ...
          'HitTest','off',...
          'Tag',sprintf('Labeler_Traj_%d',i),...
          'PickableParts','none');

        obj.hTrx(i,1) = plot(ax,...
          nan,nan,pref.TrxMarker);
        set(obj.hTrx(i,1),...
          'Color',pref.TrajColor,...
          'MarkerSize',pref.TrxMarkerSize,...
          'LineWidth',pref.TrxLineWidth,...
          'Tag',sprintf('Labeler_Trx_%d',i),...
          'ButtonDownFcn',@(h,evt) obj.clickTarget(h,evt,i),...
          'PickableParts','all',...
          'HitTest','on');
        if i == obj.currTarget,
          set(obj.hTrx(i,1),...
            'PickableParts','none',...
            'HitTest','off');
        end
        
%         obj.hTrxEll(i,1) = plot(ax,nan,nan,'-');
%         set(obj.hTrxEll(i,1),'HitTest','off',...
%           'Color',pref.TrajColor);
        
%         id = find(obj.trxIdPlusPlus2Idx==i)-1;
        obj.hTrxTxt(i,1) = text(nan,nan,num2str(i),'Parent',ax,...
          'Color',pref.TrajColor,...
          'Fontsize',pref.TrxIDLblFontSize,...
          'Fontweight',pref.TrxIDLblFontWeight,...
          'PickableParts','none',...
          'Tag',sprintf('Labeler_TrxTxt_%d',i));
      end
    end
    
    function setShowTrx(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrx = tf;
      obj.updateShowTrx();
    end

    function setShowOccludedBox(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showOccludedBox = tf;
    end
    
    function setShowTrxCurrTargetOnly(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrxCurrTargetOnly = tf;
      obj.updateShowTrx();
    end
    
    function setShowTrxIDLbl(obj,tf)
      assert(isscalar(tf) && islogical(tf));
      obj.showTrxIDLbl = tf;
      obj.updateShowTrx();
    end
    
    function updateShowTrx(obj)
      
      if ~obj.hasTrx
        return;
      end
            
      if obj.showTrx        
        if obj.showTrxCurrTargetOnly
          tfShow = false(obj.nTrx,1);
          tfShow(obj.currTarget) = true;
        else
          tfShow = true(obj.nTrx,1);
        end
      else
        tfShow = false(obj.nTrx,1);
      end

      set(obj.hTraj(tfShow),'Visible','on');
      set(obj.hTraj(~tfShow),'Visible','off');
      set(obj.hTrx(tfShow),'Visible','on');
      set(obj.hTrx(~tfShow),'Visible','off');
      if obj.showTrxIDLbl
        set(obj.hTrxTxt(tfShow),'Visible','on');
        set(obj.hTrxTxt(~tfShow),'Visible','off');
      else
        set(obj.hTrxTxt,'Visible','off');
      end

      obj.updateTrx();
      
    end
    
    function updateTrx(obj)
      % Update .hTrx, .hTraj based on .trx, .showTrx*, .currFrame
      
      if ~obj.hasTrx,
        return;
      end
      
      t = obj.currFrame;
      iTgtCurr = obj.currTarget;
      trxAll = obj.trx;
      nPre = obj.showTrxPreNFrm;
      nPst = obj.showTrxPostNFrm;
      pref = obj.projPrefs.Trx;
      
      if obj.showTrx        
        if obj.showTrxCurrTargetOnly
          tfShow = false(obj.nTrx,1);
          tfShow(iTgtCurr) = true;
        else
          tfShow = true(obj.nTrx,1);
        end
        
        iMov = obj.currMovie;
        PROPS = obj.gtGetSharedProps();
        npts = obj.nLabelPoints;
        p = reshape(obj.(PROPS.LPOS){iMov}(:,:,t,tfShow),2*npts,[]);
        % p is [npts x nShow]
        tfLbledShow = false(obj.nTrx,1);
        tfLbledShow(tfShow) = all(~isnan(p),1);  
      else
        tfShow = false(obj.nTrx,1);
      end
      

      
%       tfShowEll = isscalar(obj.showTrxEll) && obj.showTrxEll ...
%         && all(isfield(trxAll,{'a' 'b' 'x' 'y' 'theta'}));
   
      % update coords/positions
      %tic;
      for iTrx = 1:obj.nTrx
        trxCurr = trxAll(iTrx);
        t0 = trxCurr.firstframe;
        t1 = trxCurr.endframe;
        
        if t0<=t && t<=t1
          idx = t+trxCurr.off;
          xTrx = trxCurr.x(idx);
          yTrx = trxCurr.y(idx);
        else
          xTrx = nan;
          yTrx = nan;
        end
        set(obj.hTrx(iTrx),'XData',xTrx,'YData',yTrx);
        
        if tfShow(iTrx)
          tTraj = max(t-nPre,t0):min(t+nPst,t1); % could be empty array
          iTraj = tTraj + trxCurr.off;
          xTraj = trxCurr.x(iTraj);
          yTraj = trxCurr.y(iTraj);          
          
          if iTrx==iTgtCurr
            color = pref.TrajColorCurrent;
          elseif tfLbledShow(iTrx)
            color = [0 1 1]; % cyan, temporary hardcode
          else
            color = pref.TrajColor;
          end
          set(obj.hTraj(iTrx),'XData',xTraj,'YData',yTraj,'Color',color);
          set(obj.hTrx(iTrx),'Color',color);

          if obj.showTrxIDLbl
            dx = pref.TrxIDLblOffset;
            set(obj.hTrxTxt(iTrx),'Position',[xTrx+dx yTrx+dx 1],...
              'Color',color);
          end
        end
          
%           if tfShowEll && t0<=t && t<=t1
%             ellipsedraw(2*trxCurr.a(idx),2*trxCurr.b(idx),...
%               trxCurr.x(idx),trxCurr.y(idx),trxCurr.theta(idx),'-',...
%               'hEllipse',obj.hTrxEll(iTrx),'noseLine',true);
%           end
        %end
      end
      %fprintf('Time to update trx: %f\n',toc);
      set(obj.hTrx([1:iTgtCurr-1,iTgtCurr+1:end],1),...
        'PickableParts','all',...
        'HitTest','on');
      set(obj.hTrx(iTgtCurr,1),...
        'PickableParts','none',...
        'HitTest','off');
%       if tfShowEll
%         set(obj.hTrxEll(tfShow),'Visible','on');
%         set(obj.hTrxEll(~tfShow),'Visible','off');
%       else
%         set(obj.hTrxEll,'Visible','off');
%       end
    end
        
    function setSkeletonEdges(obj,se)
      obj.skeletonEdges = se;
      obj.lblCore.updateSkeletonEdges();
      obj.labeledpos2trkViz.initAndUpdateSkeletonEdges(se);
    end
    function setShowSkeleton(obj,tf)
      obj.showSkeleton = logical(tf);
      obj.lblCore.updateShowSkeleton();
      obj.labeledpos2trkViz.setShowSkeleton(tf);
    end
    function setFlipLandmarkMatches(obj,matches)
      obj.flipLandmarkMatches = matches;
    end
        
  end
  
  %% Labeling
  methods
    
    function labelingInit(obj,varargin)
      % Create LabelCore and call labelCore.init() based on current 
      % .labelMode, .nLabelPoints, .labelPointsPlotInfo, .labelTemplate  
      % For calibrated labelCores, can also require .currMovie, 
      % .viewCalibrationData, .vewCalibrationDataProjWide to be properly 
      % initted
            
      [lblmode,dosettemplate] = myparse(varargin,...
        'labelMode',[],... % if true, force a call to labelsUpdateNewFrame(true) at end of call. Poorly named option.
        'dosettemplate',true...
        );
      tfLblModeChange = ~isempty(lblmode);
      if tfLblModeChange
        assert(isa(lblmode,'LabelMode'));
      else
        lblmode = obj.labelMode;
      end
     
      nPts = obj.nLabelPoints;
      lblPtsPlotInfo = obj.labelPointsPlotInfo;
      lblPtsPlotInfo.Colors = obj.LabelPointColors;
      %template = obj.labelTemplate;
      
      lc = obj.lblCore;
      if ~isempty(lc)
        % AL 20150731. Possible/probable MATLAB bug. Existing LabelCore 
        % object should delete and clean itself up when obj.lblCore is 
        % reassigned below. The precise sequence of events is complex 
        % because various callbacks in uicontrols hold refs to the 
        % LabelCore (the callbacks get reassigned in lblCore.init though).
        %
        % However, without the explicit deletion in this branch, the 
        % existing lblCore did not get cleaned up, leading to two sets of 
        % label pts on axes_curr; moreover, when this occurred and the 
        % Labeler GUI was killed (with 'X' button in upper-right), I 
        % consistently got a segv on Win7/64, R2014b.
        %
        % With this explicit cleanup the segv goes away and of course the
        % extra labelpts are fixed as well.
        
        % AL: Need strategy for persisting/carrying lblCore state between
        % movies. newMovie prob doesn't need to call labelingInit.
        hideLabelsPrev = lc.hideLabels;
        if isprop(lc,'streamlined')
          streamlinedPrev = lc.streamlined;
        end
        delete(lc);
        obj.lblCore = [];
      else
        hideLabelsPrev = false;
      end
      obj.lblCore = LabelCore.createSafe(obj,lblmode);
      obj.lblCore.init(nPts,lblPtsPlotInfo);
      if hideLabelsPrev
        obj.lblCore.labelsHide();
      end
      if isprop(obj.lblCore,'streamlined') && exist('streamlinedPrev','var')>0
        obj.lblCore.streamlined = streamlinedPrev;
      end
      
      % labelmode-specific inits
      if dosettemplate && lblmode==LabelMode.TEMPLATE
        obj.labelingInitTemplate();
      end
      if obj.lblCore.supportsCalibration
        vcd = obj.viewCalibrationDataCurrent;
        if isempty(vcd)
%           warningNoTrace('Labeler:labelingInit',...
%             'No calibration data loaded for calibrated labeling.');
        else
          obj.lblCore.projectionSetCalRig(vcd);
        end
      end
      obj.labelMode = lblmode;
      
      obj.genericInitLabelPointViz('lblPrev_ptsH','lblPrev_ptsTxtH',...
        obj.gdata.axes_prev,lblPtsPlotInfo);
      
      if tfLblModeChange
        % sometimes labelcore need this kick to get properly set up
        obj.labelsUpdateNewFrame(true);
      end
    end
    
    function labelingInitTemplate(obj)
      % Call .lblCore.setTemplate based on a labeled frame in the proj
      % Uses .trx as appropriate
      [tffound,iMov,frm,iTgt,xyLbl] = obj.labelFindOneLabeledFrameEarliest();
      if tffound
        if obj.hasTrx
          trxfname = obj.trxFilesAllFullGTaware{iMov};
          movIfo = obj.movieInfoAllGTaware{iMov};
          trximov = obj.getTrx(trxfname,movIfo.nframes);
          trxI = trximov(iTgt);
          idx = trxI.off + frm;
          tt = struct('loc',[trxI.x(idx) trxI.y(idx)],...
            'theta',trxI.theta(idx),'pts',xyLbl);
        else
          tt = struct('loc',[nan nan],'theta',nan,'pts',xyLbl);
        end
        obj.lblCore.setTemplate(tt);
      else
        obj.lblCore.setRandomTemplate();
      end
    end
    
    %%% labelpos
        
    function labelPosClear(obj)
      % Clear all labels AND TAGS for current movie/frame/target
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      x = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt);
      if all(isnan(x(:)))
        % none; short-circuit set to avoid triggering .labeledposNeedsSave
      else        
        obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt) = nan;
        obj.labeledposNeedsSave = true;
      end
      
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = ts;
      obj.(PROPS.LPOSTAG){iMov}(:,iFrm,iTgt) = false;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
    end
    
    function labelPosClearI(obj,iPt)
      % Clear labels and tags for current movie/frame/target, point iPt
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      
      PROPS = obj.gtGetSharedProps();
      xy = obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt);
      if all(isnan(xy))
        % none; short-circuit set to avoid triggering .labeledposNeedsSave
      else
        obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = nan;
        obj.labeledposNeedsSave = true;
      end
      
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
      obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = false;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
    end
    
    function [tf,lpos,lpostag] = labelPosIsLabeled(obj,iFrm,iTrx)
      % For current movie. Labeled includes fullyOccluded
      %
      % tf: scalar logical
      % lpos: [nptsx2] xy coords for iFrm/iTrx
      % lpostag: [npts] logical array 
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      lpos = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTrx);
      tfnan = isnan(lpos);
      tf = any(~tfnan(:));
      if nargout>=3
        lpostag = obj.(PROPS.LPOSTAG){iMov}(:,iFrm,iTrx);
      end
    end 
    
    function tf = labelPosIsLabeledMov(obj,iMov)
      % iMov: movie index (row index into .movieFilesAll)
      %
      % tf: [nframes-for-iMov], true if any point labeled in that mov/frame

      %#%MVOK
      
      ifo = obj.movieInfoAll{iMov,1};
      nf = ifo.nframes;
      lpos = obj.labeledpos{iMov};
      lposnnan = ~isnan(lpos);
      
      tf = arrayfun(@(x)nnz(lposnnan(:,:,x,:))>0,(1:nf)');
    end
    
    function tf = labelPosIsOccluded(obj,iFrm,iTrx)
      % Here Occluded refers to "pure occluded"
      % For current movie.
      % iFrm, iTrx: optional, defaults to current
      % Note: it is permitted to call eg LabelPosSet with inf coords
      % indicating occluded
      
      iMov = obj.currMovie;
      if exist('iFrm','var')==0
        iFrm = obj.currFrame;
      end
      if exist('iTrx','var')==0
        iTrx = obj.currTarget;
      end
      PROPS = obj.gtGetSharedProps();
      lpos = obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTrx);
      tf = isinf(lpos(:,1));
    end
    
    function labelPosSet(obj,xy)
      % Set labelpos for current movie/frame/target
            
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(:,:,iFrm,iTgt) = xy;
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(:,iFrm,iTgt) = ts;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(:,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
      obj.labeledposNeedsSave = true;
    end
        
    function labelPosSetI(obj,xy,iPt)
      % Set labelpos for current movie/frame/target, point iPt
      
      assert(~any(isnan(xy(:))));
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = xy;
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosClearFramesI(obj,frms,iPt)
      xy = nan(2,1);
      obj.labelPosSetFramesI(frms,xy,iPt);      
    end
    
    function labelPosSetFramesI(obj,frms,xy,iPt)
      % Set labelpos for current movie/target to a single (constant) point
      % across multiple frames
      %
      % frms: vector of frames
      
      assert(isvector(frms));
      assert(numel(xy)==2);
      assert(isscalar(iPt));
      
      obj.trxCheckFramesLiveErr(frms);

      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,1,frms,iTgt) = xy(1);
      obj.(PROPS.LPOS){iMov}(iPt,2,frms,iTgt) = xy(2);
      obj.updateFrameTableComplete(); % above sets mutate .labeledpos{obj.currMovie} in more than just .currFrame
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(iPt,frms,iTgt) = ts;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(iPt,frms,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end

      obj.labeledposNeedsSave = true;
    end
    
    function labelPosSetFromLabeledPos2(obj)
      % copy .labeledpos2 to .labeledpos for current movie/frame/target
      
      assert(~obj.gtIsGTMode);
      
      iMov = obj.currMovie;
      if iMov>0
        frm = obj.currFrame;
        iTgt = obj.currTarget;
        lpos = obj.labeledpos2{iMov}(:,:,frm,iTgt);
        obj.labelPosSet(lpos);
      else
        warning('labeler:noMovie','No movie.');
      end
    end    
    
    function labelPosBulkClear(obj,varargin)
      % Clears ALL labels for current GT state -- ALL MOVIES/TARGETS
      
      [cleartype,gt] = myparse(varargin,...
        'cleartype','all',... % 'all'=>all movs all tgts
        ...                   % 'tgt'=>all movs current tgt
        ...                   % WARNING: 'tgt' assumes target i corresponds across all movies!!
        'gt',obj.gtIsGTMode);
      
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      nMov = obj.getnmoviesGTawareArg(gt);
      
      ts = now;
      iTgt = obj.currTarget;
      for iMov=1:nMov
        switch cleartype
          case 'all'
            obj.(PROPS.LPOS){iMov}(:) = nan;        
            obj.(PROPS.LPOSTS){iMov}(:) = ts;
            obj.(PROPS.LPOSTAG){iMov}(:) = false;
            if ~gt
              % unclear what this should be; marked-ness currently unused
              obj.labeledposMarked{iMov}(:) = false; 
            end
          case 'tgt'
            obj.(PROPS.LPOS){iMov}(:,:,:,iTgt) = nan;
            obj.(PROPS.LPOSTS){iMov}(:,:,iTgt) = ts;
            obj.(PROPS.LPOSTAG){iMov}(:,:,iTgt) = false;
            if ~gt
              % unclear what this should be; marked-ness currently unused
              obj.labeledposMarked{iMov}(:,:,iTgt) = false; 
            end            
          otherwise
            assert(false);
        end
      end
      if ~gt,
        obj.lastLabelChangeTS = ts;
      end
      
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosBulkImport(obj,xy,ts)
      % Set ALL labels for current movie/target
      %
      % xy: [nptx2xnfrm]
      
      assert(~obj.gtIsGTMode);
      if ~exist('ts','var'),
        ts = now;
      end
      
      iMov = obj.currMovie;
      lposOld = obj.labeledpos{iMov};
      szassert(xy,size(lposOld));
      obj.labeledpos{iMov} = xy;
      obj.labeledposTS{iMov}(:) = ts;
      obj.lastLabelChangeTS = max(obj.labeledposTS{iMov}(:));
      obj.labeledposMarked{iMov}(:) = true; % not sure of right treatment

      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      else
        obj.lastLabelChangeTS = ts;
      end
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosBulkImportTblMov(obj,tblFT,iMov)
      % Set labels for current movie/target from a table. GTmode supported.
      % Existing labels are NOT cleared!
      %
      % tblFT: table with fields .frm, .iTgt, .p, .tfocc. CANNOT have field
      % .mov, to avoid possible misunderstandings/bugs. This meth sets
      % labels on the *current movie only*.
      %   * tblFT.p should have size [n x nLabelPoints*2].
      %       The raster order is (fastest first): 
      %          {physical pt,view,coordinate (x vs y)}
      %   * tblFT.tfocc should be logical of size [n x nLabelPoints]
      %
      % iMov: optional scalar movie index into which labels are imported.
      %   Defaults to .currMovie.
      % 
      % No checking is done against image or crop size.
      
      tblfldscontainsassert(tblFT,{'frm' 'iTgt' 'p' 'tfocc'});
      tblfldsdonotcontainassert(tblFT,{'mov'});

      if exist('iMov','var')==0
        iMov = obj.currMovie;
      end
      assert(iMov>0);

      n = height(tblFT);
      npts = obj.nLabelPoints;
      szassert(tblFT.p,[n 2*npts]);
      szassert(tblFT.tfocc,[n npts]);
      assert(islogical(tblFT.tfocc));

      PROPS = obj.gtGetSharedProps();
      
      warningNoTrace('Existing labels NOT cleared.');
      lpos = obj.(PROPS.LPOS){iMov};
      lpostag = obj.(PROPS.LPOSTAG){iMov};
      lposTS = obj.(PROPS.LPOSTS){iMov};
      %lposMarked = obj.(PROPS.LlabeledposMarked{iMov};
      tsnow = now;
      for i=1:n % KB will vectorize this appropriately
        frm = tblFT.frm(i);
        itgt = tblFT.iTgt(i);        
        xy = Shape.vec2xy(tblFT.p(i,:));
        tfocc = tblFT.tfocc(i,:);
        assert(frm<=size(lpos,3));
        assert(itgt<=size(lpos,4));
        
        lpos(:,:,frm,itgt) = xy;
        lpostag(:,frm,itgt) = tfocc;
        lposTS(:,frm,itgt) = tsnow; % could allow specification in tblFT
        %lposMarked(:,frm,itgt) = true;
      end
      obj.(PROPS.LPOS){iMov} = lpos;
      obj.(PROPS.LPOSTAG){iMov} = lpostag;
      obj.(PROPS.LPOSTS){iMov} = lposTS;
      %obj.labeledposMarked{iMov} = lposMarked;
              
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      else
        obj.lastLabelChangeTS = tsnow;
      end
      obj.labeledposNeedsSave = true;
    end
    
    function labelPosBulkImportTbl(obj,tblFT)
      % Like labelPosBulkImportTblMov, but table may include movie 
      % fullpaths. 
      %
      % tblFT: table with fields .mov, .frm, .iTgt, .p. tblFT.mov are 
      % movie full-paths and they must match entries in 
      % obj.movieFilesAllFullGTaware *exactly*. 
      % For multiview projects, tblFT.mov must match 
      % obj.movieFilesAllFullGTaware(:,1).
      
      movs = tblFT.mov;
      mfaf1 = obj.movieFilesAllFullGTaware(:,1);
      [tf,iMov] = ismember(movs,mfaf1); % iMov are movie indices
      if ~all(tf)
        movsbad = unique(movs(~tf));
        error('Movies not found in project: %s',...
          String.cellstr2CommaSepList(movsbad));
      end
      
      iMovUn = unique(iMov);
      nMovUn = numel(iMovUn);
      for iiMovUn=1:nMovUn
        imovcurr = iMovUn(iiMovUn);
        tfmov = iMov==imovcurr;
        nrows = nnz(tfmov);
        fprintf('Importing %d rows for movie %d.\n',nrows,imovcurr);
        obj.labelPosBulkImportTblMov(...
          tblFT(tfmov,{'frm' 'iTgt' 'p' 'tfocc'}),imovcurr);
      end
    end
    
    function labelPosSetUnmarkedFramesMovieFramesUnmarked(obj,xy,iMov,frms)
      % Set all unmarked labels for given movie, frames. Newly-labeled 
      % points are NOT marked in .labeledposmark
      %
      % xy: [nptsx2xnumel(frms)xntgts]
      % iMov: scalar movie index
      % frms: frames for iMov; labels 3rd dim of xy
      
      assert(~obj.gtIsGTMode);
      
      npts = obj.nLabelPoints;
      ntgts = obj.nTargets;
      nfrmsSpec = numel(frms);
      assert(size(xy,1)==npts);
      assert(size(xy,2)==2);
      assert(size(xy,3)==nfrmsSpec);
      assert(size(xy,4)==ntgts);
      validateattributes(iMov,{'numeric'},{'scalar' 'positive' 'integer' '<=' obj.nmovies});
      nfrmsMov = obj.movieInfoAll{iMov,1}.nframes;
      validateattributes(frms,{'numeric'},{'vector' 'positive' 'integer' '<=' nfrmsMov});    
      
      lposmarked = obj.labeledposMarked{iMov};      
      tfFrmSpec = false(npts,nfrmsMov,ntgts);
      tfFrmSpec(:,frms,:) = true;
      tfSet = tfFrmSpec & ~lposmarked;
      tfSet = reshape(tfSet,[npts 1 nfrmsMov ntgts]);
      tfLPosSet = repmat(tfSet,[1 2]); % [npts x 2 x nfrmsMov x ntgts]
      tfXYSet = ~lposmarked(:,frms,:); % [npts x nfrmsSpec x ntgts]
      tfXYSet = reshape(tfXYSet,[npts 1 nfrmsSpec ntgts]);
      tfXYSet = repmat(tfXYSet,[1 2]); % [npts x 2 x nfrmsSpec x ntgts]
      obj.labeledpos{iMov}(tfLPosSet) = xy(tfXYSet);
      
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      else
        obj.lastLabelChangeTS = now;
      end
      obj.labeledposNeedsSave = true;  
      
%       for iTgt = 1:ntgts
%       for iFrm = 1:nfrmsSpec
%         f = frms(iFrm);
%         tfset = ~lposmarked(:,f,iTgt); % [npts x 1]
%         tfset = repmat(tfset,[1 2]); % [npts x 2]
%         lposFrmTgt = lpos(:,:,f,iTgt);
%         lposFrmTgt(tfset) = xy(:,:,iFrm,iTgt);
%         lpos(:,:,f,iTgt) = lposFrmTgt;
%       end
%       end
%       obj.labeledpos{iMov} = lpos;
    end
    
    function labelPosSetUnmarked(obj)
      % Clear .labeledposMarked for current movie/frame/target
      
      assert(~obj.gtIsGTMode);
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      obj.labeledposMarked{iMov}(:,iFrm,iTgt) = false;
    end
    
    function labelPosSetAllMarked(obj,val)
      % Clear .labeledposMarked for current movie, all frames/targets

      assert(~obj.gtIsGTMode);
      obj.labeledposMarked{iMov}(:) = val;
    end
        
    function labelPosSetOccludedI(obj,iPt)
      % Occluded is "pure occluded" here
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOS){iMov}(iPt,:,iFrm,iTgt) = inf;
      ts = now;
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = ts;
      if ~obj.gtIsGTMode
        obj.labeledposMarked{iMov}(iPt,iFrm,iTgt) = true;
        obj.lastLabelChangeTS = ts;
      end
      obj.labeledposNeedsSave = true;
    end
        
    function labelPosTagSetI(obj,iPt)
      % Set a single tag onto points
      %
      % iPt: can be vector
      %
      % The same tag value will be set to all elements of iPt.      
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTS){iMov}(iPt,iFrm,iTgt) = now();
      obj.(PROPS.LPOSTAG){iMov}(iPt,iFrm,iTgt) = true;
    end
    
    function labelPosTagClearI(obj,iPt)
      % iPt: can be vector
      
      iMov = obj.currMovie;
      iFrm = obj.currFrame;
      iTgt = obj.currTarget;
      if obj.gtIsGTMode
        obj.labeledposTSGT{iMov}(iPt,iFrm,iTgt) = now();
        obj.labeledpostagGT{iMov}(iPt,iFrm,iTgt) = false;
      else
        obj.labeledposTS{iMov}(iPt,iFrm,iTgt) = now();
        obj.labeledpostag{iMov}(iPt,iFrm,iTgt) = false;
      end
    end
    
    function labelPosTagSetFramesI(obj,iPt,frms)
      % Set tag for current movie/target, given pt/frames

      obj.trxCheckFramesLiveErr(frms);
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTAG){iMov}(iPt,frms,iTgt) = true;
    end
    
    function labelPosTagClearFramesI(obj,iPt,frms)
      % Clear tags for current movie/target, given pt/frames
      
      iMov = obj.currMovie;
      iTgt = obj.currTarget;
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.LPOSTAG){iMov}(iPt,frms,iTgt) = false;
    end
    
    function [tfneighbor,iFrm0,lpos0] = ...
                          labelPosLabeledNeighbor(obj,iFrm,iTrx) % obj const
      % tfneighbor: if true, a labeled neighboring frame was found
      % iFrm0: index of labeled neighboring frame, relevant only if
      %   tfneighbor is true
      % lpos0: [nptsx2] labels at iFrm0, relevant only if tfneighbor is true
      %
      % This method looks for a frame "near" iFrm for target iTrx that is
      % labeled. This could be iFrm itself if it is labeled. If a
      % neighboring frame is found, iFrm0 is not guaranteed to be the
      % closest or any particular neighboring frame although this will tend
      % to be true.      
      
      iMov = obj.currMovie;
      PROPS = obj.gtGetSharedProps();
      lposTrx = obj.(PROPS.LPOS){iMov}(:,:,:,iTrx);
      assert(isrow(obj.NEIGHBORING_FRAME_OFFSETS));
      for dFrm = obj.NEIGHBORING_FRAME_OFFSETS
        iFrm0 = iFrm + dFrm;
        iFrm0 = max(iFrm0,1);
        iFrm0 = min(iFrm0,obj.nframes);
        lpos0 = lposTrx(:,:,iFrm0);
        if ~isnan(lpos0(1))
          tfneighbor = true;
          return;
        end
      end
      
      tfneighbor = false;
      iFrm0 = nan;
      lpos0 = [];      
    end
    
    function [tffound,mIdx,frm,iTgt,xyLbl] = labelFindOneLabeledFrame(obj)
      % Find one labeled frame, any labeled frame.
      %
      % tffound: true if one was found.
      % mIdx: scalar MovieIndex
      % frm: scalar frame number
      % iTgt: etc
      % xyLbl: [npts x 2] labeled positions
      
      iMov = find(obj.movieFilesAllHaveLbls,1);
      if ~isempty(iMov)
        mIdx = MovieIndex(iMov,false);        
      else
        iMov = find(obj.movieFilesAllGTHaveLbls,1);
        if isempty(iMov)
          tffound = false;
          mIdx = [];
          frm = [];
          iTgt = [];
          xyLbl = [];
          return;
        end
        mIdx = MovieIndex(iMov,true);
      end
      
      lpos = obj.getLabeledPosMovIdx(mIdx);      
      nFrm = size(lpos,3);
      nTgt = size(lpos,4);
      for frm = 1:nFrm
        for iTgt=1:nTgt
          xyLbl = lpos(:,:,frm,iTgt);
          if any(~isnan(xyLbl(:)))
            tffound = true;
            return;
          end
        end
      end
      
      % Should never reach here
      tffound = false;
      mIdx = [];
      frm = [];
      iTgt = [];
      xyLbl = [];
      return;
    end
    
    function [tffound,iMov,frm,iTgt,xyLbl,mints] = labelFindOneLabeledFrameEarliest(obj)
      % Look only in labeledposGTaware, and look for the earliest labeled 
      % frame.
      
      lpos = obj.labeledposGTaware;
      lposts = obj.labeledposTSGTaware;
      tffound = false;

      mints = inf;
      
      for i = 1:numel(lpos)
        %islabeled = permute(all(obj.labeledposTSGTaware{i}>0,1),[2,3,1]);
        islabeled = permute(all(all(lpos{i}>0,1),2),[3,4,1,2]);
        idxlabeled = find(islabeled);
        tmp = lposts{i}(1,:,:,1);
        [mintscurr,j] = min(tmp(islabeled));
        [frmcurr,iTgtcurr] = ind2sub([size(lposts{i},2),size(lposts{i},3)],idxlabeled(j));
        if mintscurr < mints,
          frm = frmcurr;
          iTgt = iTgtcurr;
          iMov = i;
          mints = mintscurr;
          tffound = true;
        end
      end
      
      if tffound
        xyLbl = lpos{iMov}(:,:,frm,iTgt);
      else
        iMov = [];
        frm = [];
        iTgt = [];
        xyLbl = [];
      end
    end
    
    function [nTgts,nPts] = labelPosLabeledFramesStats(obj,frms) % obj const
      % Get stats re labeled frames in the current movie.
      % 
      % frms: vector of frame indices to consider. Defaults to
      %   1:obj.nframes.
      %
      % nTgts: numel(frms)-by-1 vector indicating number of targets labeled
      %   for each frame in consideration
      % nPts: numel(frms)-by-1 vector indicating number of points labeled 
      %   for each frame in consideration, across all targets
      
      if exist('frms','var')==0
        if isnan(obj.nframes)
          frms = [];
        else
          frms = 1:obj.nframes;
        end
        tfWaitBar = true;
      else
        tfWaitBar = false;
      end
      
      if ~obj.hasMovie || obj.currMovie==0 % invariants temporarily broken
        nTgts = nan(numel(frms),1);
        nPts = nan(numel(frms),1);
        return;
      end
      
      nf = numel(frms);
      ntgts = obj.nTargets;
      lpos = obj.labeledposCurrMovie;
      tflpos = ~isnan(lpos); % true->labeled (either regular or occluded)      
      
      nTgts = zeros(nf,1);
      nPts = zeros(nf,1);
      if tfWaitBar
        hWB = waitbar(0,'Updating frame table');
        centerOnParentFigure(hWB,obj.gdata.figure);
        ocp = onCleanup(@()delete(hWB));
        tic;
      end
      for i = 1:nf
        if tfWaitBar && toc >= .25
          waitbar(i/nf,hWB);
          tic;
        end
        f = frms(i);
        
        % don't squeeze() here it's expensive        
        tmpNTgts = 0;
        tmpNPts = 0;
        for iTgt = 1:ntgts
          z = sum(tflpos(:,1,f,iTgt));
          tmpNPts = tmpNPts+z;
          tfTgtLabeled = (z>0);
          if tfTgtLabeled
            tmpNTgts = tmpNTgts+1;
          end
        end
        nTgts(i) = tmpNTgts;
        nPts(i) = tmpNPts;        
      end
    end
    
    function tf = labelPosMovieHasLabels(obj,iMov,varargin)
      gt = myparse(varargin,'gt',obj.gtIsGTMode);
      if ~gt
        lpos = obj.labeledpos{iMov};
      else
        lpos = obj.labeledposGT{iMov};
      end
      tf = any(~isnan(lpos(:)));
    end
    
    % Label Cosmetics notes 20190601
    %
    % - Cosmetics settings live in PV pairs on .labelPointsPlotInfo
    % - lblCore owns a subset of these PVs in lblCore.ptsPlotInfo. This is
    % a copy (the copyness is an impl detail). lblCore could prob just look 
    % at lObj.labelPointsPlotInfo but having a copy isn't all bad and can 
    % be advantageous at times. A copy of the cosmetics state needs to 
    % exist outside of HG properties in the various handle objects b/c 
    % lblCore subclasses can mutate handle cosmetics in various ways (eg 
    % due to selection, adjustment, occludedness, etc) and these mutations 
    % need to be revertable to the baseline setpoint.
    % - During project save, getCurrentConfig includes .labelPointsPlotInfo
    % to be saved. 
    % - During project load, .labelPointsPlotInfo can be modernized to
    % include/remove fields per the default config (.DEFAULT_CFG_FILENAME)
    %
    % - View> Hide/Show labels applies only to the mainUI (lblCore), and
    % applies to both markers and text. Note that text visibility is also
    % independently toggleable.
    
    % Prediction Cosmetics notes
    % 
    % - Cosmetics live in PV pairs on .predPointsPlotInfo
    % - These props include: Color, Marker-related, Text-related
    % - The .Color prop currently is *per-set*, ie corresponding points in
    % different views currently must have the same color;
    % size(pppi.Color,1) will equal .nPhysPoints. See PredictPointColors().
    % - .pppi applies to:
    %   i) *All* trackers. Currently LabelTrackers all contain a
    %   TrackingVisualizer for visualization. All trackers have their TVs
    %   initted from pppi and thus all trackers' tracking results currently
    %   will look the same. Nothing stops the user from individually
    %   massaging a trackers' TV, but for now there is no facility to 
    %   display preds from multiple tracker objs at the same time, and
    %   any such changes are not currently serialized.
    %   iii) PPPI serves as an initialization point for aux tracking 
    %   results in .trkResViz, but it is expected that the user will mutate
    %   the cosmetics for .trkResViz to facilitate comparison of multiple
    %   tracking results.
    %      a) User-mutations of .trkResViz cosmetic state IS SERIALIZED
    %      with the project. Note in contrast, TrackingVisualizers for
    %      LabelTrackers (in i) currently are NOT serialized at all.
    %
    % Use cases.
    %  1. Comparison across multiple "live" tracking results. 
    %     - Currently this can't really be done, since results from
    %     multiple LabelTrackers cannot be displayed simultaneously, not to
    %     mention that all LabelTrackers share the same cosmetics.
    %  2. Comparison between one "live" tracking result and one imported.
    %     - This is at least possible, but imported results share the same 
    %     cosmetics as "live" results so it's not great.
    %  3. Comparison between multiple imported tracking results.
    %     - This works well as the .trkRes* infrastruture is explicitly 
    %     designed for this (currently cmdline only).
    %
    % Future 20190603.
    %  A. The state of 1. above is unclear as we currently do not even save
    %  tracking results with the project at all. In the meantime, use case
    %  3. does meet the basic need provided tracking results are first
    %  exported. This also serves more general purposes eg when a single
    %  tracker is run across a parameter sweep, or with differing training
    %  data etc.
    %  B. Re 2. above, imported results quite likely should just be a 
    %  special case of the aux (.trkRes*) tracking results. This would
    %  simplify the code a bit, cosmetics would be mutable, and cosmetics
    %  settings would be saved with the project.
    
    function updateLandmarkColors(obj,colorSpecs)
      for i=1:numel(colorSpecs)
        cs = colorSpecs(i);
        lsetType = cs.landmarkSetType;
        lObjUpdateMeth = lsetType.updateColorLabelerMethod();
        obj.(lObjUpdateMeth)(cs.colors,cs.colormapname);
      end
    end
    
    function updateLandmarkCosmetics(obj,mrkrSpecs)
      for i=1:numel(mrkrSpecs)
        ms = mrkrSpecs(i);
        lsetType = ms.landmarkSetType;
        lObjUpdateMeth = lsetType.updateCosmeticsLabelerMethod();
        obj.(lObjUpdateMeth)(ms.MarkerProps,ms.TextProps,ms.TextOffset);
      end
    end
    
    function updateLandmarkLabelColors(obj,colors,colormapname)
      % colors: "setwise" colors

      szassert(colors,[obj.nPhysPoints 3]);
      lc = obj.lblCore;
      % Colors apply to lblCore, lblPrev_*, timeline
      
      obj.labelPointsPlotInfo.ColorMapName = colormapname;
      obj.labelPointsPlotInfo.Colors = colors;
      ptcolors = obj.Set2PointColors(colors);
      lc.updateColors(ptcolors);
      LabelCore.setPtsColor(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH,ptcolors);
      obj.gdata.labelTLInfo.updateLandmarkColors();
    end
    
    function updateLandmarkPredictionColors(obj,colors,colormapname)
      % colors: "setwise" colors
      szassert(colors,[obj.nPhysPoints 3]);
      
      obj.predPointsPlotInfo.Colors = colors;
      obj.predPointsPlotInfo.ColorMapName = colormapname;
      tAll = obj.trackersAll;
      for i=1:numel(tAll)
        if ~isempty(tAll{i})
          tAll{i}.updateLandmarkColors();
        end
      end      
      %obj.gdata.labelTLInfo.updateLandmarkColors();
    end
    
    function updateLandmarkImportedColors(obj,colors,colormapname)
      % colors: "setwise" colors
      szassert(colors,[obj.nPhysPoints 3]);
      
      obj.impPointsPlotInfo.Colors = colors;
      obj.impPointsPlotInfo.ColorMapName = colormapname;
      lpos2tv = obj.labeledpos2trkViz;
      ptcolors = obj.Set2PointColors(colors);
      lpos2tv.updateLandmarkColors(ptcolors);
      for i=1:numel(obj.trkResViz)
        obj.trkResViz{i}.updateLandmarkColors(ptcolors);
      end
    end

    function updateLandmarkLabelCosmetics(obj,pvMarker,pvText,textOffset)

      lc = obj.lblCore;
      
      % Marker cosmetics apply to lblCore, lblPrev_*
      flds = fieldnames(pvMarker);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.labelPointsPlotInfo.MarkerProps.(f) = pvMarker.(f);
      end
      lc.updateMarkerCosmetics(pvMarker);      
      set(obj.lblPrev_ptsH,pvMarker);
      
      % Text cosmetics apply to lblCore, lblPrev_*
      flds = fieldnames(pvText);
      for f=flds(:)',f=f{1}; %#ok<FXSET>
        obj.labelPointsPlotInfo.TextProps.(f) = pvText.(f);
      end
      obj.labelPointsPlotInfo.TextOffset = textOffset;
      set(obj.lblPrev_ptsTxtH,pvText);
      obj.prevAxesLabelsRedraw(); % should use .TextOffset
      lc.updateTextLabelCosmetics(pvText,textOffset);
      %obj.labelsUpdateNewFrame(true); % should redraw prevaxes too
    end
    function [tfHideTxt,pvText] = hlpUpdateLandmarkCosmetics(obj,...
        pvMarker,pvText,ptsPlotInfoFld)
      % set PVs on .ptsPlotInfo field; mild massage
      
      fns = fieldnames(pvMarker);
      for f=fns(:)',f=f{1}; %#ok<FXSET> 
        % this allows pvMarker to be 'incomplete'; could just set entire
        % struct
        obj.(ptsPlotInfoFld).MarkerProps.(f) = pvMarker.(f);
      end
      fns = fieldnames(pvText);
      for f=fns(:)',f=f{1}; %#ok<FXSET>
        obj.(ptsPlotInfoFld).TextProps.(f) = pvText.(f);
      end
      % TrackingVisualizer wants this prop broken out
      tfHideTxt = strcmp(pvText.Visible,'off'); % could make .Visible field optional 
      pvText = rmfield(pvText,'Visible');
    end 
    function updateLandmarkPredictionCosmetics(obj,pvMarker,pvText,textOffset)
      [tfHideTxt,pvText] = obj.hlpUpdateLandmarkCosmetics(...
        pvMarker,pvText,'predPointsPlotInfo');
      tAll = obj.trackersAll;
      for i=1:numel(tAll)
        if ~isempty(tAll{i})
          tv = tAll{i}.trkVizer;
          tv.setMarkerCosmetics(pvMarker);
          tv.setTextCosmetics(pvText);
          tv.setTextOffset(textOffset);
          tv.setHideTextLbls(tfHideTxt);
        end
      end      
    end
    
    function updateLandmarkImportedCosmetics(obj,pvMarker,pvText,textOffset)
       [tfHideTxt,pvText] = obj.hlpUpdateLandmarkCosmetics(...
        pvMarker,pvText,'impPointsPlotInfo');      
      
      lpos2tv = obj.labeledpos2trkViz;
      lpos2tv.setMarkerCosmetics(pvMarker);      
      lpos2tv.setTextCosmetics(pvText);
      lpos2tv.setTextOffset(textOffset);
      lpos2tv.setHideTextLbls(tfHideTxt);
      
      % Todo, set on .trkRes*
    end

  end
  
  methods (Static)
    function trkfile = genTrkFileName(rawname,sMacro,movfile)
      % Generate a trkfilename from rawname by macro-replacing.      
      [sMacro.movdir,sMacro.movfile] = fileparts(movfile);
      trkfile = FSPath.macroReplace(rawname,sMacro);
      if ~(numel(rawname)>=4 && strcmp(rawname(end-3:end),'.trk'))
        trkfile = [trkfile '.trk'];
      end
    end
    function [tfok,trkfiles] = checkTrkFileNamesExportUI(trkfiles,varargin)
      % Check/confirm trkfile names for export. If any trkfiles exist, ask 
      % whether overwriting is ok; alternatively trkfiles may be 
      % modified/uniqueified using datetimestamps.
      %
      % trkfiles (input): cellstr of proposed trkfile names (full paths).
      % Can be an array.
      %
      % tfok: if true, trkfiles (output) is valid, and user has said it is 
      % ok to write to those files even if it is an overwrite.
      % trkfiles (output): cellstr, same size as trkfiles. .trk filenames
      % that are okay to write/overwrite to. Will match input if possible.
      
      noUI = myparse(varargin,...
        'noUI',false);
      
      tfexist = cellfun(@(x)exist(x,'file')>0,trkfiles(:));
      tfok = true;
      if any(tfexist)
        iExist = find(tfexist,1);
        queststr = sprintf('One or more .trk files already exist, eg: %s.',trkfiles{iExist});
        if noUI
          btn = 'Add datetime to filenames';
          warningNoTrace('Labeler:trkFileNamesForExport',...
            'One or more .trk files already exist. Adding datetime to trk filenames.');
        else
          btn = questdlg(queststr,'Files exist','Overwrite','Add datetime to filenames',...
            'Cancel','Add datetime to filenames');
        end
        if isempty(btn)
          btn = 'Cancel';
        end
        switch btn
          case 'Overwrite'
            % none; use trkfiles as-is
          case 'Add datetime to filenames'
            nowstr = datestr(now,'yyyymmddTHHMMSS');
            [trkP,trkF] = cellfun(@fileparts,trkfiles,'uni',0);            
            trkfiles = cellfun(@(x,y)fullfile(x,[y '_' nowstr '.trk']),trkP,trkF,'uni',0);
          otherwise
            tfok = false;
            trkfiles = [];
        end
      end
    end
  end

  methods
    
    function sMacro = baseTrkFileMacros(obj)
      % Set of built-in macros appropriate to exporting files/data
      sMacro = struct();
      sMacro.projname = obj.projname;
      if ~isempty(obj.projectfile)
        [sMacro.projdir,sMacro.projfile] = fileparts(obj.projectfile);
      else
        sMacro.projdir = '';
        sMacro.projfile = '';
      end
      tObj = obj.tracker;
      if ~isempty(tObj)
        sMacro.trackertype = tObj.algorithmName;
      else
        sMacro.trackertype = 'undefinedtracker';
      end
    end
    
    function trkfile = defaultTrkFileName(obj,movfile)
      % Only client is GMMTracker
      trkfile = Labeler.genTrkFileName(obj.defaultExportTrkRawname(),...
        obj.baseTrkFileMacros(),movfile);
    end
    
    function rawname = defaultExportTrkRawname(obj,varargin)
      % Default raw/base trkfilename for export (WITH macros, NO extension).
      
      labels = myparse(varargin,...
        'labels',false... % true for eg manual labels (as opposed to automated tracking)
        );
      
      if ~isempty(obj.projectfile)
        basename = '$movfile_$projfile';
      elseif ~isempty(obj.projname)
        basename = '$movfile_$projname';
      else
        basename = '$movfile';
      end
      
      if labels
        gt = obj.gtIsGTMode;
        if gt
          basename = [basename '_gtlabels'];
        else
          basename = [basename '_labels'];
        end
      else
        basename = [basename '_$trackertype'];
      end
      
      rawname = fullfile('$movdir',basename);
    end
    
    function [tfok,rawtrkname] = getExportTrkRawnameUI(obj,varargin)
      % Prompt the user to get a raw/base trkfilename.
      %
      % varargin: see defaultExportTrkRawname
      % 
      % tfok: user canceled or similar
      % rawtrkname: use only if tfok==true
      
      rawtrkname = inputdlg('Enter name/pattern for trkfile(s) to be exported. Available macros: $movdir, $movfile, $projdir, $projfile, $projname, $trackertype.',...
        'Export Trk File',1,{obj.defaultExportTrkRawname(varargin{:})});
      tfok = ~isempty(rawtrkname);
      if tfok
        rawtrkname = rawtrkname{1};
      end
    end
    
    function fname = getDefaultFilenameExportStrippedLbl(obj)
      lblstr = 'TrainData';
      if ~isempty(obj.projectfile)
        rawname = ['$projdir/$projfile_' lblstr '.mat'];
      elseif ~isempty(obj.projname)
        rawname = ['$projdir/$projname_' lblstr '.mat'];
      else
        rawname = ['$projdir/' lblstr datestr(now,'yyyymmddTHHMMSS') '.mat'];
      end
      sMacro = obj.baseTrkFileMacros();
      fname = FSPath.macroReplace(rawname,sMacro);
    end
    
    function fname = getDefaultFilenameExportLabelTable(obj)
      if obj.gtIsGTMode
        lblstr = 'gtlabels';
      else
        lblstr = 'labels';
      end
      if ~isempty(obj.projectfile)
        rawname = ['$projdir/$projfile_' lblstr '.mat'];
      elseif ~isempty(obj.projname)
        rawname = ['$projdir/$projname_' lblstr '.mat'];
      else
        rawname = ['$projdir/' lblstr '.mat'];
      end
      sMacro = obj.baseTrkFileMacros();
      fname = FSPath.macroReplace(rawname,sMacro);
    end
        
    function [tfok,trkfiles] = getTrkFileNamesForExportUI(obj,movfiles,...
        rawname,varargin)
      % Concretize a raw trkfilename, then check for conflicts etc.
      
      noUI = myparse(varargin,...
        'noUI',false);
      
      sMacro = obj.baseTrkFileMacros();
      trkfiles = cellfun(@(x)Labeler.genTrkFileName(rawname,sMacro,x),...
        movfiles,'uni',0);
      [tfok,trkfiles] = Labeler.checkTrkFileNamesExportUI(trkfiles,'noUI',noUI);
    end
    
    function [tfok,trkfiles] = resolveTrkfilesVsTrkRawname(obj,iMovs,...
        trkfiles,rawname,defaultRawNameArgs,varargin)
      % Ugly, input arg helper. Methods that export a trkfile must have
      % either i) the trkfilenames directly supplied, ii) a raw/base 
      % trkname supplied, or iii) nothing supplied.
      %
      % If i), check the sizes.
      % If ii), generate the trkfilenames from the rawname.
      % If iii), first generate the rawname, then generate the
      % trkfilenames.
      %
      % Cases ii) and iii), are also UI/prompt if there are
      % existing/conflicting filenames already on disk.
      %
      % defaultRawNameArgs: cell of PVs to pass to defaultExportTrkRawname.
      % 
      % iMovs: vector, indices into .movieFilesAllGTAware
      %
      % tfok: scalar, if true then trkfiles is usable; if false then user
      %   canceled or similar.
      % trkfiles: [iMovs] cellstr, trkfiles (full paths) to export to
      % 
      % This call can also throw.
      
      noUI = myparse(varargin,...
        'noUI',false);
      
      PROPS = obj.gtGetSharedProps();
      
      movfiles = obj.(PROPS.MFAF)(iMovs,:);
      if isempty(trkfiles)
        if isempty(rawname)
          rawname = obj.defaultExportTrkRawname(defaultRawNameArgs{:});
        end
        [tfok,trkfiles] = obj.getTrkFileNamesForExportUI(movfiles,...
          rawname,'noUI',noUI);
        if ~tfok
          return;
        end
      end
      
      nMov = numel(iMovs);
      nView = obj.nview;
      if size(trkfiles,1)~=nMov
        error('Labeler:argSize',...
          'Numbers of movies and trkfiles supplied must be equal.');
      end
      if size(trkfiles,2)~=nView
        error('Labeler:argSize',...
          'Number of columns in trkfiles (%d) must equal number of views in project (%d).',...
          size(trkfiles,2),nView);
      end
      
      tfok = true;
    end
  end
  
  methods
	
    function [trkfilesCommon,kwCommon,trkfilesAll] = ...
                      getTrkFileNamesForImport(obj,movfiles)
      % Find available trkfiles for import
      %
      % movfiles: cellstr of movieFilesAllFull
      %
      % trkfilesCommon: [size(movfiles)] cell-of-cellstrs.
      % trkfilesCommon{i} is a cellstr with numel(kwCommon) elements.
      % trkfilesCommon{i}{j} contains the jth common trkfile found for
      % movfiles{i}, where j indexes kwCommon.
      % kwCommon: [nCommonKeywords] cellstr of common keywords found.
      % trkfilesAll: [size(movfiles)] cell-of-cellstrs. trkfilesAll{i}
      % contains all trkfiles found for movfiles{i}, a superset of
      % trkfilesCommon{i}.
      %
      % "Common" trkfiles are those that share the same naming pattern
      % <moviename>_<keyword>.trk and are present for all movfiles.
      
      trkfilesAll = cell(size(movfiles));
      keywords = cell(size(movfiles));
      for i=1:numel(movfiles)
        mov = movfiles{i};
        if exist(mov,'file')==0
          error('Labeler:noMovie','Cannot find movie: %s.',mov);
        end
        
        [movdir,movname] = fileparts(mov);
        trkpat = [movname '*.trk'];
        dd = dir(fullfile(movdir,trkpat));
        trkfilesAllShort = {dd.name}';
        trkfilesAll{i} = cellfun(@(x)fullfile(movdir,x),trkfilesAllShort,'uni',0);
        
        trkpatRE = sprintf('%s(?<kw>.*).trk',movname);
        re = regexp(trkfilesAllShort,trkpatRE,'names');
        re = cellfun(@(x)x.kw,re,'uni',0);
        keywords{i} = re;
        
        assert(numel(trkfilesAll{i})==numel(keywords{i}));
      end
      
      % Find common keywords
      kwUn = unique(cat(1,keywords{:}));
      tfKwUnCommon = cellfun(@(zKW) all(cellfun(@(x)any(strcmp(x,zKW)),keywords(:))),kwUn);
      kwCommon = kwUn(tfKwUnCommon);
      trkfilesCommon = cellfun(@(zTFs,zKWs) cellfun(@(x)zTFs(strcmp(zKWs,x)),kwCommon), ...
        trkfilesAll,keywords,'uni',0);
    end
    
    function labelExportTrkGeneric(obj,iMovs,trkfiles,...
        lposFld,lposTSFld,lposTagFld)
      % For export given lpos* fields for iMovs into trkfiles. 
      %
      % lposTSFld, lposTagFld: can be empty
      %
      % The GT-status of obj is irrelevant, iMovs just indexes lposFld*.
      
      tfTS = ~isempty(lposTSFld);
      tfTag = ~isempty(lposTagFld);
      
      nMov = numel(iMovs);
      nView = obj.nview;
      nPhysPts = obj.nPhysPoints;
      for i=1:nMov
        iMvSet = iMovs(i);
        lposFull = obj.(lposFld){iMvSet};
        if tfTS
          lposTSFull = obj.(lposTSFld){iMvSet};
        end
        if tfTag
          lposTagFull = obj.(lposTagFld){iMvSet};
        end
        
        for iView=1:nView
          iPt = (1:nPhysPts) + (iView-1)*nPhysPts;
          if nView==1
            assert(nPhysPts==size(lposFull,1));
          else
            tmp = find(obj.labeledposIPt2View==iView);
            assert(isequal(tmp(:),iPt(:)));
          end
          
          args = cell(1,0);
          if tfTS
            args = [args {'pTrkTS' lposTSFull(iPt,:,:)}]; %#ok<AGROW>
          end
          if tfTag
            args = [args {'pTrkTag' lposTagFull(iPt,:,:)}]; %#ok<AGROW>
          end

          trkfile = TrkFile(lposFull(iPt,:,:,:),args{:});
          trkfile.save(trkfiles{i,iView});
          fprintf('Saved trkfile: %s\n',trkfiles{i,iView});
        end
      end
      msgbox(sprintf('Results for %d moviesets exported.',nMov),'Export complete');      
    end
    
    function labelExportTrk(obj,iMovs,varargin)
      % Export label data to trk files.
      %
      % iMov: optional, indices into (rows of) .movieFilesAllGTaware to 
      %   export. Defaults to 1:obj.nmoviesGTaware.
      
      [trkfiles,rawtrkname] = myparse(varargin,...
        'trkfiles',[],... % [nMov nView] cellstr, fullpaths to trkfilenames to export to
        'rawtrkname',[]... % string, rawname to apply over iMovs to generate trkfiles
        );
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmoviesGTaware;
      end
      
      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovs,trkfiles,...
        rawtrkname,{'labels' true});
      if ~tfok
        return;
      end
      
      PROPS = obj.gtGetSharedProps;
      obj.labelExportTrkGeneric(iMovs,trkfiles,...
        PROPS.LPOS,PROPS.LPOSTS,PROPS.LPOSTAG);        
    end
    
    function labelImportTrkGeneric(obj,iMovSets,trkfiles,lposFld,...
                                            lposTSFld,lposTagFld)
      % Import (iMovSets,trkfiles) into the specified .labeledpos* fields
      %
      % iMovStes: [N] vector of movie set indices
      % trkfiles: [Nxnview] cellstr of trk filenames
      % lpos*Fld: property names for labeledpos, labeledposTS,
      % labeledposTag. Can be empty to not set that prop.
      
      nMovSets = numel(iMovSets);
      szassert(trkfiles,[nMovSets obj.nview]);
      nPhysPts = obj.nPhysPoints;   
      tfMV = obj.isMultiView;
      nView = obj.nview;
      
      tfTS = ~isempty(lposTSFld);
      tfTag = ~isempty(lposTagFld);
      
      for i=1:nMovSets
        iMov = iMovSets(i);
        lpos = nan(size(obj.(lposFld){iMov}));
        if tfTS
          lposTS = -inf(size(obj.(lposTSFld){iMov}));
        end
        if tfTag
          lpostag = false(size(obj.(lposTagFld){iMov}));
        end
        assert(size(lpos,1)==nPhysPts*nView);
        
        if tfMV
          fprintf('MovieSet %d...\n',iMov);
        end
        for iVw = 1:nView
          tfile = trkfiles{i,iVw};
          s = load(tfile,'-mat');
          s = TrkFile.modernizeStruct(s);
          
          if isfield(s,'pTrkiPt')
            iPt = s.pTrkiPt;
          else
            iPt = 1:size(s.pTrk,1);
          end
          tfInBounds = 1<=iPt & iPt<=nPhysPts;
          if any(~tfInBounds)
            if tfMV
              error('Labeler:trkImport',...
                'View %d: trkfile contains information for more points than exist in project (number physical points=%d).',...
                iVw,nPhysPts);
            else
              error('Labeler:trkImport',...
                'Trkfile contains information for more points than exist in project (number of points=%d).',...
                nPhysPts);
            end
          end
          if nnz(tfInBounds)<nPhysPts
            if tfMV
              warningNoTrace('Labeler:trkImport',...
                'View %d: trkfile does not contain labels for all points in project (number physical points=%d).',...
                iVw,nPhysPts);              
            else
               warningNoTrace('Labeler:trkImport',...
                 'Trkfile does not contain information for all points in project (number of points=%d).',...
                 nPhysPts);
            end
          end
          
          if isfield(s,'pTrkFrm')
            frmsTrk = s.pTrkFrm;
          else
            frmsTrk = 1:size(s.pTrk,3);
          end
          
          nfrmLpos = size(lpos,3);
          tfInBounds = 1<=frmsTrk & frmsTrk<=nfrmLpos;
          if any(~tfInBounds)
            warningNoTrace('Labeler:trkImport',...
              'Trkfile contains information for frames beyond end of movie (number of frames=%d). Ignoring additional frames.',...
              nfrmLpos);
          end
          if nnz(tfInBounds)<nfrmLpos
            warningNoTrace('Labeler:trkImport',...
              'Trkfile does not contain information for all frames in movie. Frames missing from Trkfile will be unlabeled.');
          end
          frmsTrkIB = frmsTrk(tfInBounds);
          
          if isfield(s,'pTrkiTgt')
            iTgt = s.pTrkiTgt;
          else
            iTgt = 1;
          end
          assert(size(s.pTrk,4)==numel(iTgt));
          nTgtProj = size(lpos,4);
          tfiTgtIB = 1<=iTgt & iTgt<=nTgtProj;
          if any(~tfiTgtIB)
            warningNoTrace('Labeler:trkImport',...
              'Trkfile contains information for targets not present in movie. Ignoring extra targets.');
          end
          if nnz(tfiTgtIB)<nTgtProj
            warningNoTrace('Labeler:trkImport',...
              'Trkfile does not contain information for all targets in movie.');
          end
          iTgtsIB = iTgt(tfiTgtIB);
          
          fprintf(1,'Loaded %d frames for %d points, %d targets from trk file:\n  %s.\n',...
            numel(frmsTrkIB),numel(iPt),numel(iTgtsIB),tfile);
        
          %displaying when .trk file was last updated
          tfileDir = dir(tfile);
          disp(['  trk file last modified: ',tfileDir.date])

          iPt = iPt + (iVw-1)*nPhysPts;
          lpos(iPt,:,frmsTrkIB,iTgtsIB) = s.pTrk(:,:,tfInBounds,tfiTgtIB);
          if tfTS
            lposTS(iPt,frmsTrkIB,iTgtsIB) = s.pTrkTS(:,tfInBounds,tfiTgtIB);
          end
          if tfTag
            lpostag(iPt,frmsTrkIB,iTgtsIB) = s.pTrkTag(:,tfInBounds,tfiTgtIB);
          end
        end

        obj.(lposFld){iMov} = lpos;
        if tfTS
          obj.(lposTSFld){iMov} = lposTS;
        end
        if tfTag
          obj.(lposTagFld){iMov} = lpostag;
        end
      end      
    end
    
    function labelImportTrk(obj,iMovs,trkfiles)
      % Import label data from trk files.
      %
      % iMovs: [nMovie]. Movie(set) indices for which to import.
      % trkfiles: [nMoviexnview] cellstr. full filenames to trk files
      %   corresponding to iMov.
      
      assert(~obj.gtIsGTMode);
      
      obj.labelImportTrkGeneric(iMovs,trkfiles,'labeledpos',...
        'labeledposTS','labeledpostag');
      % need to compute lastLabelChangeTS from scratch
      obj.computeLastLabelChangeTS();
      
      obj.movieFilesAllHaveLbls(iMovs) = ...
        cellfun(@Labeler.computeNumLbledRows,obj.labeledpos(iMovs));
      
      obj.updateFrameTableComplete();
      if obj.gtIsGTMode
        obj.gtUpdateSuggMFTableLbledComplete('donotify',true);
      end
      
      %obj.labeledposNeedsSave = true; AL 20160609: don't touch this for
      %now, since what we are importing is already in the .trk file.
      obj.labelsUpdateNewFrame(true);
      
      RC.saveprop('lastTrkFileImported',trkfiles{end});
    end
    
    % compute lastLabelChangeTS from scratch
    function computeLastLabelChangeTS(obj)
      
      obj.lastLabelChangeTS = max(cellfun(@(x) max(x(:)),obj.labeledposTS));
      
    end
    
    function [tfsucc,trkfilesUse] = labelImportTrkFindTrkFilesPrompt(obj,movfiles)
      % Find trkfiles present for given movies. Prompt user to pick a set
      % if more than one exists.
      %
      % movfiles: [nTrials x nview] cellstr
      %
      % tfsucc: if true, trkfilesUse is valid; if false, trkfilesUse is
      % intedeterminate
      % trkfilesUse: cellstr, same size as movfiles. Full paths to trkfiles
      % present/selected for import
      
      [trkfilesCommon,kwCommon] = obj.getTrkFileNamesForImport(movfiles);
      nCommon = numel(kwCommon);
      
      tfsucc = false;
      trkfilesUse = [];
      switch nCommon
        case 0
          warningNoTrace('Labeler:labelImportTrkPrompt',...
            'No consistently-named trk files found across %d given movies.',numel(movfiles));
          return;
        case 1
          trkfilesUseIdx = 1;
        otherwise
          msg = sprintf('Multiple consistently-named trkfiles found. Select trkfile pattern to import.');
          uiwait(msgbox(msg,'Multiple trkfiles found','modal'));
          trkfileExamples = trkfilesCommon{1};
          for i=1:numel(trkfileExamples)
            [~,trkfileExamples{i}] = myfileparts(trkfileExamples{i});
          end
          [sel,ok] = listdlg(...
            'Name','Select trkfiles',...
            'Promptstring','Select a trkfile (pattern) to import.',...
            'SelectionMode','single',...
            'listsize',[300 300],...
            'liststring',trkfileExamples);
          if ok
            trkfilesUseIdx = sel;
          else
            return;
          end
      end
      trkfilesUse = cellfun(@(x)x{trkfilesUseIdx},trkfilesCommon,'uni',0);
      tfsucc = true;
    end
    
    % 20180628 iss 202.
    % The following chain of "smart" methods
    % labelImportTrkPromptGenericAuto
    %  labelImportTrkPromptAuto
    %  labels2ImprotTrkPromptAuto
    %
    % were trying too hard for more vanilla use cases, in particular
    % single-view single-movie import situs.
    %
    % However, they may still still be useful for bulk use cases:
    % multiview, or large-numbers-of-movies. So leave them around for now.
    % 
    % Currently these this chain has only a SINGLE CALLER: importTrkResave,
    % which operates in a bulk fashion.
    %
    % The new simplified call is just 
    % labelImportTrkPromptGenericSimple
    % which is called by LabelerGUI for single-movieset situs (including
    % multiview)
    
    function labelImportTrkPromptGenericAuto(obj,iMovs,importFcn)
      % Come up with trkfiles based on iMovs and then call importFcn.
      % 
      % iMovs: index into .movieFilesAllGTAware
      
      PROPS = obj.gtGetSharedProps();
      movfiles = obj.(PROPS.MFAF)(iMovs,:);
      [tfsucc,trkfilesUse] = obj.labelImportTrkFindTrkFilesPrompt(movfiles);
      if tfsucc
        feval(importFcn,obj,iMovs,trkfilesUse);
      else
        if isscalar(iMovs) && obj.nview==1
          % In this case (single movie, single view) failure can occur if 
          % no trkfile is found alongside movie, or if user cancels during
          % a prompt.
          
          lastTrkFileImported = RC.getprop('lastTrkFileImported');
          if isempty(lastTrkFileImported)
            lastTrkFileImported = pwd;
          end
          [fname,pth] = uigetfile('*.trk','Import trkfile',lastTrkFileImported);
          if isequal(fname,0)
            return;
          end
          trkfile = fullfile(pth,fname);
          feval(importFcn,obj,iMovs,{trkfile});
        end
      end      
    end
	
    function labelImportTrkPromptGenericSimple(obj,iMov,importFcn,varargin)
      % Prompt user for trkfiles to import and import them with given 
      % importFcn. User can cancel to abort
      %
      % iMov: scalar positive index into .movieFilesAll. GT mode not
      %   allowed.
      
      gtok = myparse(varargin,...
        'gtok',false ... % if true, obj.gtIsGTMode can be true, and iMov 
                  ...% refers per GT state. importFcn needs to support GT
                  ...% state
                  );
      
      assert(isscalar(iMov));      
      if ~gtok
        assert(~obj.gtIsGTMode);
      end
      
      movs = obj.movieFilesAllFullGTaware(iMov,:);
      movdirs = cellfun(@fileparts,movs,'uni',0);
      nvw = obj.nview;
      trkfiles = cell(1,nvw);
      for ivw=1:nvw
        if nvw>1
          promptstr = sprintf('Import trkfile for view %d',ivw);
        else
          promptstr = 'Import trkfile';
        end
        [fname,pth] = uigetfile('*.trk',promptstr,movdirs{ivw});
        if isequal(fname,0)
          return;
        end
        trkfiles{ivw} = fullfile(pth,fname);
      end
      
      feval(importFcn,obj,iMov,trkfiles);
    end
    
    function labelImportTrkPromptAuto(obj,iMovs)
      % Import label data from trk files, prompting if necessary to specify
      % which trk files to import.
      %
      % iMovs: [nMovie]. Optional, movie(set) indices to import.
      %
      % labelImportTrkPrompt will look for trk files with common keywords
      % (consistent naming) in .movieFilesAllFull(iMovs). If there is
      % precisely one consistent trkfile pattern, it will import those
      % trkfiles. Otherwise it will ask the user which trk files to import.
      
      assert(~obj.gtIsGTMode);

      if exist('iMovs','var')==0
        iMovs = 1:obj.nmovies;
      end
      obj.labelImportTrkPromptGenericAuto(iMovs,'labelImportTrk');
    end
    
    function labelMakeLabelMovie(obj,fname,varargin)
      % Make a movie of all labeled frames for current movie
      %
      % fname: output filename, movie to be created.
      % optional pvs:
      % - framerate. defaults to 10.
      
      [frms2inc,framerate] = myparse(varargin,...
        'frms2inc','all',... % 
        'framerate',10 ...
      );

      if ~obj.hasMovie
        error('Labeler:noMovie','No movie currently open.');
      end
      if exist(fname,'file')>0
        error('Labeler:movie','Output movie ''%s'' already exists. For safety reasons, this movie will not be overwritten. Please specify a new output moviename.',...
          fname);
      end
      
      switch frms2inc
        case 'all'
          frms = 1:obj.nframes;
        case 'lbled'
          nTgts = obj.labelPosLabeledFramesStats();
          frms = find(nTgts>0);
          if nFrms==0
            msgbox('Current movie has no labeled frames.');
            return;
          end
        otherwise
          assert(false);
      end

      nFrms = numel(frms);

      ax = obj.gdata.axes_curr;
      axlims = axis(ax);
      vr = VideoWriter(fname); 
      vr.FrameRate = framerate;

      vr.open();
      try
        hTxt = text(230,10,'','parent',obj.gdata.axes_curr,'Color','white','fontsize',24);
        hWB = waitbar(0,'Writing video');
        for i = 1:nFrms
          f = frms(i);
          obj.setFrame(f);
          axis(ax,axlims);
          hTxt.String = sprintf('%04d',f);
          tmpFrame = getframe(ax);
          vr.writeVideo(tmpFrame);
          waitbar(i/nFrms,hWB,sprintf('Wrote frame %d\n',f));
        end
      catch ME
        vr.close();
        delete(hTxt);
        ME.rethrow();
      end
      vr.close();
      delete(hTxt);
      delete(hWB);
    end   
    
    function tblMF = labelGetMFTableLabeled(obj,varargin)
      % Compile mov/frm/tgt MFTable; include all labeled frames/tgts. 
      %
      % Includes nonGT/GT rows per current GT state.
      %
      % Can return [] indicating "no labels of requested/specified type"
      %
      % tblMF: See MFTable.FLDSFULLTRX.
      
      [wbObj,useLabels2,useMovNames,tblMFTrestrict,useTrain,tfMFTOnly] = myparse(varargin,...
        'wbObj',[], ... % optional WaitBarWithCancel. If cancel:
                   ... % 1. obj logically const (eg except for obj.trxCache)
                   ... % 2. tblMF (output) indeterminate
        'useLabels2',false,... % if true, use labels2 instead of labels
        'useMovNames',false,... % if true, use movieNames instead of movieIndices
        'tblMFTrestrict',[],... % if supplied, tblMF is the labeled subset 
                           ... % of tblMFTrestrict (within fields .mov, 
                           ... % .frm, .tgt). .mov must be a MovieIndex.
                           ... % tblMF ordering should be as in tblMFTrestrict
        'useTrain',[],... % whether to use training labels (1) gt labels (0), or whatever current mode is ([])
        'MFTOnly',false... % if true, only return mov, frm, target
        ); 
      tfWB = ~isempty(wbObj);
      tfRestrict = ~isempty(tblMFTrestrict);
      
      if useLabels2
        if isempty(useTrain)
          mfts = MFTSetEnum.AllMovAllLabeled2;
        elseif useTrain
          mfts = MFTSet(MovieIndexSetVariable.AllTrnMov,FrameSetVariable.Labeled2Frm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
        else % useGT
          mfts = MFTSet(MovieIndexSetVariable.AllGTMov,FrameSetVariable.Labeled2Frm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
        end
      else
        if isempty(useTrain)
          mfts = MFTSetEnum.AllMovAllLabeled;
        elseif useTrain
          mfts = MFTSet(MovieIndexSetVariable.AllTrnMov,FrameSetVariable.LabeledFrm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);          
        else % useGT
          mfts = MFTSet(MovieIndexSetVariable.AllGTMov,FrameSetVariable.LabeledFrm,...
                        FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
        end
      end
      tblMF = mfts.getMFTable(obj);
      
      if tfRestrict
        tblMF = MFTable.intersectID(tblMF,tblMFTrestrict);
      end
      
      if tfMFTOnly,
        return;
      end
      
      if isequal(tblMF,[]) % this would have errored below in call to labelAddLabelsMFTableStc
        return;
      end
      
      if obj.hasTrx,
        if isempty(useTrain),
          trxFiles = obj.trxFilesAllFullGTaware;
        elseif useTrain == 0,
          trxFiles = obj.trxFilesAllGTFull;
        else
          trxFiles = obj.trxFilesAllFull;
        end
          
        argsTrx = {'trxFilesAllFull',trxFiles,...
          'trxCache',obj.trxCache};
      else
        argsTrx = {};
      end
      if useLabels2
        
        if isempty(useTrain),
          lpos = obj.labeledpos2GTaware;
          lpostag = obj.labeledpostagGTaware;
          lposTS = obj.labeledposTSGTaware;
        elseif useTrain == 0,
          lpos = obj.labeledpos2GT;
          lpostag = obj.labeledpostagGT;
          lposTS = obj.labeledposTSGT;
        else
          lpos = obj.labeledpos2;
          lpostag = obj.labeledpostag;
          lposTS = obj.labeledposTS;
        end
        lpostag = cellfun(@(x)false(size(x)),lpostag,'uni',0);
        lposTS = cellfun(@(x)-inf(size(x)),lposTS,'uni',0);
        
      else
        
        if isempty(useTrain),
          lpos = obj.labeledposGTaware;
          lpostag = obj.labeledpostagGTaware;
          lposTS = obj.labeledposTSGTaware;
        elseif useTrain == 0,
          lpos = obj.labeledposGT;
          lpostag = obj.labeledpostagGT;
          lposTS = obj.labeledposTSGT;
        else
          lpos = obj.labeledpos;
          lpostag = obj.labeledpostag;
          lposTS = obj.labeledposTS;
        end
        
      end
      
      tblMF = Labeler.labelAddLabelsMFTableStc(tblMF,lpos,lpostag,lposTS,...
          argsTrx{:},'wbObj',wbObj);
      if tfWB && wbObj.isCancel
        % tblMF (return) indeterminate
        return;
      end
      
      if useMovNames
        assert(isa(tblMF.mov,'MovieIndex'));
        tblMF.mov = obj.getMovieFilesAllFullMovIdx(tblMF.mov);
      end
    end
    
    %#%GTOK
    function tblMF = labelGetMFTableCurrMovFrmTgt(obj)
      % Get MFTable for current movie/frame/target (single-row table)
      %
      % tblMF: See MFTable.FLDSFULLTRX.
                  
      if obj.gtIsGTMode
        % Easy to support in GT mode, just unnec for now
        error('Labeler:gt','Not supported in GT mode.');
      end
      
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      lposFrmTgt = obj.labeledpos{iMov}(:,:,frm,iTgt);
      lpostagFrmTgt = obj.labeledpostag{iMov}(:,frm,iTgt);
      lposTSFrmTgt = obj.labeledposTS{iMov}(:,frm,iTgt);      

      mov = iMov;
      p = Shape.xy2vec(lposFrmTgt); % absolute position
      pTS = lposTSFrmTgt';
      tfocc = lpostagFrmTgt';
      if obj.hasTrx
        assert(~obj.isMultiView,'Unsupported for multiview.');
        assert(obj.frm2trx(frm,iTgt));
        trxCurr = obj.trx(iTgt);
        xtrxs = trxCurr.x(frm+trxCurr.off);
        ytrxs = trxCurr.y(frm+trxCurr.off);
        sclrassert(xtrxs); % legacy check
        sclrassert(ytrxs);
        pTrx = [xtrxs ytrxs];
      else
        pTrx = [nan nan];
      end
      
      tblMF = table(mov,frm,iTgt,p,pTS,tfocc,pTrx);
    end
    
    function tblMF = labelMFTableAddROITrx(obj,tblMF,roiRadius,varargin)
      % Add .pRoi and .roi to tblMF using trx info
      %
      % tblMF.pRoi: Just like tblMF.p, but relative to tblMF.roi (p==1 => 
      %   first row/col of ROI)
      % tblMF.roi: [nrow x (2*2*nview)]. Raster order {lo,hi},{x,y},view      
      
      [rmOOB,pfld,proifld] = myparse(varargin,...
        'rmOOB',true,... % if true, remove rows where shape is outside roi
        'pfld','p',...  % optional "source" lbl field
        'proifld','pRoi'... % optional "dest" roi-lbl field
        );
      
      %tblfldscontainsassert(tblMF,MFTable.FLDSFULLTRX);
      % no requirements beyond as accessed in code
      
      tblfldsdonotcontainassert(tblMF,{proifld 'roi'});
      
      nphyspts = obj.nPhysPoints;
      nrow = height(tblMF);
      p = tblMF.(pfld);
      pTrx = tblMF.pTrx;
      xy = Shape.vecs2xys(p.'); % [npts x 2 x nrow]
      xyTrx = Shape.vecs2xys(pTrx.'); % [ntrxpts x 2 x nrow]
      
      [roi,tfOOBview,xyRoi] = Shape.xyAndTrx2ROI(xy,xyTrx,nphyspts,roiRadius);
      % roi: [nrow x 4*nview]
      % tfOOBview: [nrow x nview]
      % xyRoi: [npts x 2 x nrow]
      npts = obj.nLabelPoints;
      szassert(xyRoi,[npts 2 nrow]);
      pRoi = reshape(xyRoi,npts*2,nrow).'; % [nrow x D]
      
      tblMF = [tblMF table(pRoi,roi,'VariableNames',{proifld 'roi'})];

      if rmOOB
        tfRmrow = any(tfOOBview,2);
        nrm = nnz(tfRmrow);
        if nrm>0
          warningNoTrace('Labeler:oob',...
            '%d rows with shape out of bounds of target ROI. These rows will be discarded.',nrm);
          tblMF(tfRmrow,:) = [];
        end
      end
                    
%       pRoi = nan(size(p));
%       roi = nan(nrow,4*obj.nview);
%       for i=1:nrow
%         xy = Shape.vec2xy(p(i,:));
%         xyTrx = Shape.vec2xy(pTrx(i,:));
%          [roiCurr,tfOOBview,xyROIcurr] = ...
%            Shape.xyAndTrx2ROI(xy,xyTrx,nphyspts,roiRadius);
%         if rmOOB && any(tfOOBview)
%           warningNoTrace('CPRLabelTracker:oob',...
%             'Movie(set) %d, frame %d, target %d: shape out of bounds of target ROI. Not including row.',...
%             tblMF.mov(i),tblMF.frm(i),tblMF.iTgt(i));
%           tfRmRow(i) = true;
%         else
%           pRoi(i,:) = Shape.xy2vec(xyROIcurr);
%           roi(i,:) = roiCurr;
%         end
%       end
    end
    
    function tblMF = labelMFTableAddROICrop(obj,tblMF,varargin)
      % Add .pRoi and .roi to tblMF using crop info
      %
      % tblMF.pRoi: Just like tblMF.p, but relative to tblMF.roi (p==1 => 
      %   first row/col of ROI)
      % tblMF.roi: [nrow x (2*2*nview)]. Raster order {lo,hi},{x,y},view
      %
      % tblMF(out): rows removed if xy are OOB of roi.
      
      [rmOOB,pfld,proifld] = myparse(varargin,...
        'rmOOB',true,... % if true, remove rows where shape is outside roi
        'pfld','p',...  % optional "source" lbl field
        'proifld','pRoi'... % optional "dest" roi-lbl field
        );
      
      %tblfldscontainsassert(tblMF,MFTable.FLDSFULL);
      % no requirements beyond as accessed in code
      tblfldsdonotcontainassert(tblMF,{proifld 'roi'});
      assert(isa(tblMF.mov,'MovieIndex'));
     
      if ~obj.cropProjHasCrops
        error('Project does not contain cropping information.');
      end
      
      obj.cropCheckCropSizeConsistency();
      
      nphyspts = obj.nPhysPoints;
      nvw = obj.nview;
      n = height(tblMF);
      p = tblMF.(pfld);      
      tfRmRow = false(n,1); % true to rm row due to OOB
      pRoi = nan(size(p));
      roi = nan(n,4*obj.nview);
      for i=1:n
        mIdx = tblMF.mov(i);
        cInfo = obj.getMovieFilesAllCropInfoMovIdx(mIdx);
        roiCurr = cat(2,cInfo.roi);
        szassert(roiCurr,[1 4*nvw]);
        
        % See Shape.p2pROI etc
        xy = Shape.vec2xy(p(i,:));
        [xyROI,tfOOBview] = Shape.xy2xyROI(xy,roiCurr,nphyspts);
        if ~rmOOB,
          tfOOBview(:) = false;
        end
        if any(tfOOBview)
          warningNoTrace('CPRLabelTracker:oob',...
            'Movie(set) %d, frame %d, target %d: shape out of bounds of target ROI. Not including row.',...
            mIdx,tblMF.frm(i),tblMF.iTgt(i));
          tfRmRow(i) = true;
        else
          pRoi(i,:) = Shape.xy2vec(xyROI);
          roi(i,:) = roiCurr;
        end
      end
      
      tblMF = [tblMF table(pRoi,roi,'VariableNames',{proifld 'roi'})];
      tblMF(tfRmRow,:) = [];
    end
    
    function tblMF = labelAddLabelsMFTable(obj,tblMF,varargin)
      mIdx = tblMF.mov;
      assert(isa(mIdx,'MovieIndex'));
      [~,gt] = mIdx.get();
      assert(all(gt) || all(~gt),...
        'Currently only all-GT or all-nonGT supported.');
      gt = gt(1);
      PROPS = Labeler.gtGetSharedPropsStc(gt);
      if obj.hasTrx
        tfaf = obj.(PROPS.TFAF);
      else
        tfaf = [];
      end
      tblMF = Labeler.labelAddLabelsMFTableStc(tblMF,...
        obj.(PROPS.LPOS),obj.(PROPS.LPOSTAG),obj.(PROPS.LPOSTS),...
        'trxFilesAllFull',tfaf,'trxCache',obj.trxCache,varargin{:});
    end
    
    function hFgs = labelOverlayMontage(obj,varargin)
      [trxCtred,trxCtredRotAlignMeth,roiRadius,roiPadVal,hFig0] = myparse(varargin,...
        'trxCtred',false,... % If true, center shapes relative to trx.x, trx.y
        'trxCtredRotAlignMeth','none',... % Rotational alignment method when trxCentered=true. One of {'none','headtail','trxtheta'}. 
        ... % 'trxCtredSizeNorm',false,... True to normalize shapes by trx.a, trx.b. SKIP THIS for now. Have found that doing this normalization tightens shape distributions a bit (when tracking/trx is good)
        'roiRadius',nan,... % A little unusual, used if .preProcParams.TargetCrop.Radius is not avail
        'roiPadVal',0,...% A little unsuual, used if .preProcParams.TargetCrop.PadBkgd is not avail
        'hFig0',[]... % Optional, previous figure to use with figurecascaded
        ); 

      if ~obj.hasMovie
        error('Please open a movie first.');
      end
      if trxCtred && ~obj.hasTrx
        error('Project does not have trx. Cannot perform trx-centered montage.');
      end
      if obj.cropProjHasCrops
        error('Currently unsupported for projects with cropping.');
      end
      
      
      nvw = obj.nview;
      nphyspts = obj.nPhysPoints;
      vwNames = obj.viewNames;
      mfts = MFTSetEnum.AllMovAllLabeled;
      tMFT = mfts.getMFTable(obj); % if GT, should get all GT labeled rows
      tMFT = obj.labelAddLabelsMFTable(tMFT);
            
      [ims,p] = obj.hlpOverlayMontageGenerateImP(tMFT,nphyspts,...
        trxCtred,trxCtredRotAlignMeth,roiRadius,roiPadVal);
      n = size(p,1);
      % p is [n x nphyspts*nvw*2]
      p = reshape(p',[nphyspts nvw 2 n]);
      
      % KB 20181022 - removing references to ColorsSets
      clrs = obj.labelPointsPlotInfo.Colors;
      ec = OlyDat.ExperimentCoordinator;      

      tbases = cell(nvw,1);
      hFgs = gobjects(nvw,1);
      hAxs = gobjects(nvw,1);
      hIms = gobjects(nvw,1);
      clckHandlers = OlyDat.XYPlotClickHandler.empty(0,1);
      hLns = gobjects(nvw,nphyspts); % line/plot handles
      for ivw=1:nvw
        if ivw==1
          if ~isempty(hFig0)
            hFgs(ivw) = figurecascaded(hFig0);
          else
            hFgs(ivw) = figure;
          end
        else
          hFgs(ivw) = figurecascaded(hFgs(1));
        end        
        hAxs(ivw) = axes;
        hIms(ivw) = imshow(ims{ivw});
        hIms(ivw).PickableParts = 'none';
        set(hIms(ivw),'Tag',sprintf('image_LabelOverlayMontage_vw%d',ivw));
        caxis auto
        hold on;
%         axis xy;
        set(hAxs(ivw),'XTick',[],'YTick',[],'Visible','on');
        if trxCtred
          switch trxCtredRotAlignMeth
            case 'none'
              rotStr = 'Unaligned';
            case 'headtail'
              rotStr = 'Head/tail aligned';
            case 'trxtheta'
              rotStr = 'Trx/theta aligned';
          end
        else
          rotStr = '';
        end
          
        if nvw>1
          tstr = sprintf('View: %s. %d labeled frames.',...
            vwNames{ivw},height(tMFT));
        else
          tstr = sprintf('%d labeled frames.',height(tMFT));
        end
        if ~isempty(rotStr)
          tstr = sprintf('%s %s.',tstr,rotStr);
        end
        title(tstr,'fontweight','bold');
        tbases{ivw} = tstr;
        
        xall = squeeze(p(:,ivw,1,:)); % [npts x nfrm]
        yall = squeeze(p(:,ivw,2,:)); % [npts x nfrm]
        eids = repmat(1:height(tMFT),nphyspts,1);
        clckHandlers(ivw,1) = OlyDat.XYPlotClickHandler(hAxs(ivw),xall(:),yall(:),eids(:),ec,false);
        
        for ipts=1:nphyspts
          x = squeeze(p(ipts,ivw,1,:));
          y = squeeze(p(ipts,ivw,2,:));
          hP = plot(hAxs(ivw),x,y,'.','markersize',4,'color',clrs(ipts,:));
          hP.PickableParts = 'none';
          hLns(ivw,ipts) = hP;
        end
        
        hCM = uicontextmenu('parent',hFgs(ivw),'Tag',sprintf('LabelOverlayMontages_vw%d',ivw));
        uimenu('Parent',hCM,'Label','Clear selection',...
          'Separator','on',...
          'Callback',@(src,evt)ec.sendSignal([],zeros(0,1)),...
          'Tag',sprintf('LabelOverlayMontage_vw%d_ClearSelection',ivw));
        uimenu('Parent',hCM,'Label','Navigate APT to selected frame',...
          'Callback',@(s,e)hlpOverlayMontage(obj,clckHandlers(1),tMFT,s,e),...
          'Tag',sprintf('LabelOverlayMontage_vw%d_NavigateToSelectedFrame',ivw)); 
        % Need only one clickhandler; the first is set up here
        set(hAxs(ivw),'UIContextMenu',hCM);
      end

      for ivw=1:nvw
        hCM = hAxs(ivw).UIContextMenu;
        hM1 = uimenu('Parent',hCM,'Label','Increase marker size',...
          'Callback',@(src,evt)obj.hlpOverlayMontageMarkerInc(hLns,2),...
          'Tag',sprintf('LabelOverlayMontage_vw%d_IncreaseMarkerSize',ivw));
        hM2 = uimenu('Parent',hCM,'Label','Decrease marker size',...
          'Callback',@(src,evt)obj.hlpOverlayMontageMarkerInc(hLns,-2),...
          'Tag',sprintf('LabelOverlayMontage_vw%d_DecreaseMarkerSize',ivw));
        uistack(hM2,'bottom');
        uistack(hM1,'bottom');
      end
      
      tor = TrainingOverlayReceiver(hAxs,tbases,tMFT);
      ec.registerObject(tor,'respond');    
    end
    function hlpOverlayMontage(obj,clickHandler,tMFT,src,evt)
      eid = clickHandler.fSelectedEids;
      if ~isempty(eid)
        trow = tMFT(eid,:);
        obj.setMFT(trow.mov,trow.frm,trow.iTgt);
      else
        warningNoTrace('No shape selected.');
      end
    end
    function [ims,p] = hlpOverlayMontageGenerateImP(obj,tMFT,nphyspts,...
        trxCtred,trxCtredRotAlignMeth,roiRadius,roiPadVal)
      % Generate images and shapes to plot
      %
      % tMFT: table with labeled frames
      % trxCtred: If true, labels will be shifted to be relative to their
      %  trx centers. If false, labels may/will wander over the image if/as
      %  targets wander
      % trxCtredRotAlignMeth: One of {'none','headtail','trxtheta'}:
      %  * 'none'. labels/shapes are not rotated. 
      %  * 'headtail'. shapes are aligned based on their iHead/iTail
      %  pts (taken from tracking parameters)
      %  * 'trxtheta'. shapes are aligned based on their trx.theta. If the
      %  trx.theta is incorrect then the alignment will be as well.
      % roiRadius:
      % roiPadVal:
      % 
      % ims: [nview] cell array of images to plot
      % p: all labels [nlbledfrm x D==(nphyspts*nvw*d)]      
      
      nvw = obj.nview;
      
      ims = obj.gdata.images_all;
      ims = arrayfun(@(x)x.CData,ims,'uni',0);
      if trxCtred
        ppParams = obj.preProcParams;
        if isempty(ppParams)
          warningNoTrace('Preprocessing parameters unset. Using supplied/default ROI radius and background pad value.');
          if ~isnan(roiRadius)
            % OK; user-supplied
          else
            [nr1,nc1] = size(ims{1});
            roiRadius = min(floor(nr1/2),floor(nc1/2)); % b/c ... why not
          end
          % roiPadVal has been supplied
        else
          % Override roiRadius, roiPadVal with .preProcParams stuff
          roiRadius = ppParams.TargetCrop.Radius;
          roiPadVal = ppParams.TargetCrop.PadBkgd;
        end

        % Image: use image for current mov/frm/tgt
        assert(nvw==1,'Expect single view for projects with trx.');
        [xTrxCurrTgt,yTrxCurrTgt,thTrxCurrTgt] = ...
          readtrx(obj.trx,obj.currFrame,obj.currTarget);
        xTrxCurrTgt = double(xTrxCurrTgt);
        yTrxCurrTgt = double(yTrxCurrTgt);
        thTrxCurrTgt = double(thTrxCurrTgt);
        switch trxCtredRotAlignMeth
          case 'none'
            % im: crop around current target, no rotation
            [roiXloCurrTgt,roiXhiCurrTgt,roiYloCurrTgt,roiYhiCurrTgt] = ...
              xyRad2roi(xTrxCurrTgt,yTrxCurrTgt,roiRadius);
            ims{1} = padgrab(ims{1},roiPadVal,...
              roiYloCurrTgt,roiYhiCurrTgt,roiXloCurrTgt,roiXhiCurrTgt); % asserted nvw==1
          case {'headtail' 'trxtheta'}
            % im: cropped + canonically rotated
            im = ims{1};
            [imnr,imnc] = size(im);
            xim = 1:imnc;
            yim = 1:imnr;
            [xgim,ygim] = meshgrid(xim,yim);
            xroictr = -roiRadius:roiRadius;
            yroictr = -roiRadius:roiRadius;
            [xgroi,ygroi] = meshgrid(xroictr,yroictr);
            im = readpdf2(double(im),xgim,ygim,xgroi,ygroi,...
              xTrxCurrTgt,yTrxCurrTgt,thTrxCurrTgt);
            ims{1} = im;
          otherwise
            assert(false);
        end
                
        % p (Shapes)
        p = tMFT.p; % [nLbld x nphyspts*(nvw==1)*2]
        pTrx = tMFT.pTrx; % [nLbld x 2]        
        n = size(p,1);
        switch trxCtredRotAlignMeth
          case 'none'
            % AL20200522: this can all be optimized now, see eg 
            % .xyAndTrx2ROI vectorized api and .labelMFTableAddROITrx
            for i=1:n
              xyRow = Shape.vec2xy(p(i,:));
              xyTrxRow = Shape.vec2xy(pTrx(i,:));

              [~,tfOOB,xyRoi] = Shape.xyAndTrx2ROI(xyRow,xyTrxRow,...
                nphyspts,roiRadius);
              if tfOOB
                trow = tMFT(i,:);
                warningNoTrace('Shape (mov %d,frm %d,tgt %d) falls outside ROI.',...
                  trow.mov,trow.frm,trow.iTgt);
              end
              p(i,:) = Shape.xy2vec(xyRoi);
            end
          case {'headtail' 'trxtheta'}
            % Add pTrx as (nphyspts+1)th point, we will use it to center
            % our aligned shapes
            pWithTrx = [p(:,1:nphyspts)     pTrx(:,1) ...
                        p(:,nphyspts+1:end) pTrx(:,2)]; 
            if strcmp(trxCtredRotAlignMeth,'headtail')
%               tObj = obj.tracker;
%               if ~isempty(tObj) && strcmp(tObj.algorithmName,'cpr')
%                 iptHead = tObj.sPrm.Reg.rotCorrection.iPtHead;
%                 iptTail = tObj.sPrm.Reg.rotCorrection.iPtTail;
%               else
              sPrm = obj.trackGetParams;
              sPrmRotCorr = sPrm.ROOT.CPR.RotCorrection;
              iptHead = sPrmRotCorr.HeadPoint;
              iptTail = sPrmRotCorr.TailPoint;
                %error('Cannot use head-tail alignment method; no tracking rotational correction settings available.');
              pWithTrxAligned = Shape.alignOrientationsOrigin(pWithTrx,iptHead,iptTail); 
              % aligned based on iHead/iTailpts, now with arbitrary offset
              % b/c was rotated about origin. Note the presence of pTrx as
              % the "last" point should not affect iptHead/iptTail defns
              
            else % 'trxtheta'
              thTrx = tMFT.thetaTrx;
              pWithTrxAligned = Shape.rotate(pWithTrx,-thTrx,[0 0]); % could rotate about pTrx but shouldn't matter
              % aligned based on trx.theta, now with arbitrary offset
            end

            twoRadP1 = 2*roiRadius+1;
            for i=1:n
              xyRowWithTrx = Shape.vec2xy(pWithTrxAligned(i,:));
              xyRowWithTrx = bsxfun(@minus,xyRowWithTrx,xyRowWithTrx(end,:)); 
              % subtract off pTrx. All pts/coords now relative to origin at
              % pTrx, with shape aligned.
              xyRow = xyRowWithTrx(1:end-1,:);
              xyRow(:,1) = xyRow(:,1)+roiRadius+1;
              xyRow(:,2) = xyRow(:,2)+roiRadius+1;
              tfOOB = xyRow<1 | xyRow>twoRadP1; % [nphyspts x 2]
              if any(tfOOB(:))
                trow = tMFT(i,:);
                warningNoTrace('Shape (mov %d,frm %d,tgt %d) falls outside ROI.',...
                  trow.mov,trow.frm,trow.iTgt);
              end
              p(i,:) = Shape.xy2vec(xyRow); % in-place modification of p
            end                        
          otherwise
            assert(false,'Unrecognized ''trxCtredRotAlignMeth''.');
        end
      else
        % ims: no change
        p = tMFT.p;
      end
    end
    function hlpOverlayMontageMarkerInc(obj,hLns,dSz) %#ok<INUSL>
      sz = hLns(1).MarkerSize;
      sz = max(sz+dSz,1);
      [hLns.MarkerSize] = deal(sz);
    end
  end
  
  methods (Static)
    function n = computeNumLbledRows(lpos)
      % Mirrors comp in updateFrameTable*
      tf = ~isnan(lpos); % labeled, est-occ, or fully-occ
      %[npts,d,nfrm,ntgt] = size(lpos);
      x = any(tf(:,1,:,:),1); % x-coords; this is done in updateFrameTable*
      n = nnz(x);
    end
%     function tblMF = labelGetMFTableLabeledStc(lpos,lpostag,lposTS,...
%         trxFilesAllFull,trxCache)
%       % Compile MFtable, by default for all labeled mov/frm/tgts
%       %
%       % tblMF: [NTrl rows] MFTable, one row per labeled movie/frame/target.
%       %   MULTIVIEW NOTE: tbl.p* is the 2d/projected label positions, ie
%       %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
%       %   index, 2. view index, 3. coord index (x vs y)
%       %   
%       %   Fields: {'mov' 'frm' 'iTgt' 'p' 'pTS' 'tfocc' 'pTrx'}
%       %   Here 'p' is 'pAbs' or absolute position
%                   
%       s = structconstruct(MFTable.FLDSFULLTRX,[0 1]);
%       
%       nMov = size(lpos,1);
%       nView = size(trxFilesAllFull,2);
%       szassert(lpos,[nMov 1]);
%       szassert(lpostag,[nMov 1]);
%       szassert(lposTS,[nMov 1]);
%       szassert(trxFilesAllFull,[nMov nView]);
%       
%       for iMov = 1:nMov
%         lposI = lpos{iMov};
%         lpostagI = lpostag{iMov};
%         lposTSI = lposTS{iMov};
%         [npts,d,nfrms,ntgts] = size(lposI);
%         assert(d==2);
%         szassert(lpostagI,[npts nfrms ntgts]);
%         szassert(lposTSI,[npts nfrms ntgts]);
%         
%         [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
%                                   trxCache,trxFilesAllFull(iMov,:),nfrms);        
%         cellfun(@(x)assert(numel(x)==ntgts),trxI);        
%         for f=1:nfrms
%           for iTgt=1:ntgts
%             lposIFrmTgt = lposI(:,:,f,iTgt);
%             % read if any point (in any view) is labeled for this
%             % (frame,target)
%             tfReadTgt = any(~isnan(lposIFrmTgt(:)));
%             if tfReadTgt
%               assert(frm2trxTotAnd(f,iTgt),'Labeled target is not live.');
%               lpostagIFrmTgt = lpostagI(:,f,iTgt);
%               lposTSIFrmTgt = lposTSI(:,f,iTgt);
%               xtrxs = cellfun(@(xx)xx(iTgt).x(f+xx(iTgt).off),trxI);
%               ytrxs = cellfun(@(xx)xx(iTgt).y(f+xx(iTgt).off),trxI);
%               
%               s(end+1,1).mov = iMov; %#ok<AGROW> 
%               s(end).frm = f;
%               s(end).iTgt = iTgt;
%               s(end).p = Shape.xy2vec(lposIFrmTgt);
%               s(end).pTS = lposTSIFrmTgt';
%               s(end).tfocc = strcmp(lpostagIFrmTgt','occ');
%               s(end).pTrx = [xtrxs(:)' ytrxs(:)'];
%             end
%           end
%         end
%       end
%       tblMF = struct2table(s,'AsArray',true);      
%     end
    
    function tblMF = labelAddLabelsMFTableStc(tblMF,lpos,lpostag,lposTS,...
        varargin)
      % Add label/trx information to an MFTable
      %
      % tblMF (input): MFTable with flds MFTable.FLDSID. tblMF.mov are 
      %   MovieIndices. tblMF.mov.get() are indices into lpos,lpostag,lposTS.
      % lpos...lposTS: as in labelGetMFTableLabeledStc
      %
      % tblMF (output): Same rows as tblMF, but with addnl label-related
      %   fields as in labelGetMFTableLabeledStc
      
      [trxFilesAllFull,trxCache,wbObj] = myparse(varargin,...
        'trxFilesAllFull',[],... % cellstr, indexed by tblMV.mov. if supplied, tblMF will contain .pTrx field
        'trxCache',[],... % must be supplied if trxFilesAllFull is supplied
        'wbObj',[]... % optional WaitBarWithCancel. If cancel, tblMF (output) indeterminate
        );      
      tfWB = ~isempty(wbObj);
      
      assert(istable(tblMF));
      tblfldscontainsassert(tblMF,MFTable.FLDSID);
      nMov = size(lpos,1);
      szassert(lpos,[nMov 1]);
      szassert(lpostag,[nMov 1]);
      szassert(lposTS,[nMov 1]);
      
      tfTrx = ~isempty(trxFilesAllFull);
      if tfTrx
        nView = size(trxFilesAllFull,2);
        szassert(trxFilesAllFull,[nMov nView]);
        tfTfafEmpty = cellfun(@isempty,trxFilesAllFull);
        % Currently, projects allowed to have some movs with trxfiles and
        % some without.
        assert(all( all(tfTfafEmpty,2) | all(~tfTfafEmpty,2) ),...
          'Unexpected trxFilesAllFull specification.');
        tfMovHasTrx = all(~tfTfafEmpty,2); % tfTfafMovEmpty(i) indicates whether movie i has trxfiles        
      else
        nView = 1;
      end
  
      nrow = height(tblMF);
      
      if tfWB
        wbObj.startPeriod('Compiling labels','shownumden',true,...
          'denominator',nrow);
        oc = onCleanup(@()wbObj.endPeriod);
        wbtime = tic;
        maxwbtime = .1; % update waitbar every second
      end
      
      % Maybe Optimize: group movies together

      npts = size(lpos{1},1);
      
      pAcc = nan(nrow,npts*2);
      pTSAcc = -inf(nrow,npts);
      tfoccAcc = false(nrow,npts);
      pTrxAcc = nan(nrow,nView*2); % xv1 xv2 ... xvk yv1 yv2 ... yvk
      thetaTrxAcc = nan(nrow,nView);
      aTrxAcc = nan(nrow,nView);
      bTrxAcc = nan(nrow,nView);
      tfInvalid = false(nrow,1); % flags for invalid rows of tblMF encountered
      iMovsAll = tblMF.mov.get;
      frmsAll = tblMF.frm;
      iTgtAll = tblMF.iTgt;
      
      iMovsUnique = unique(iMovsAll);
      nRowsComplete = 0;
      
      for movIdx = 1:numel(iMovsUnique),
        iMov = iMovsUnique(movIdx);
        rowsCurr = find(iMovsAll == iMov); % absolute row indices into tblMF
        
        lposI = lpos{iMov};
        lpostagI = lpostag{iMov};
        lposTSI = lposTS{iMov};
        [npts,d,nfrms,ntgts] = size(lposI);
        assert(d==2);
        szassert(lpostagI,[npts nfrms ntgts]);
        szassert(lposTSI,[npts nfrms ntgts]);
        
        if tfTrx && tfMovHasTrx(iMov)
          [trxI,~,frm2trxTotAnd] = Labeler.getTrxCacheAcrossViewsStc(...
            trxCache,trxFilesAllFull(iMov,:),nfrms);
          
          assert(isscalar(trxI),'Multiview projs with trx currently unsupported.');
          trxI = trxI{1};
        end
        
        for jrow = 1:numel(rowsCurr),
          irow = rowsCurr(jrow); % absolute row index into tblMF
          
          if tfWB && toc(wbtime) >= maxwbtime,
            wbtime = tic;
            tfCancel = wbObj.updateFracWithNumDen(nRowsComplete);
            if tfCancel
              return;
            end
          end
          
          %tblrow = tblMF(irow,:);
          frm = frmsAll(irow);
          iTgt = iTgtAll(irow);
          
          if frm<1 || frm>nfrms
            tfInvalid(irow) = true;
            continue;
          end
          
          if tfTrx && tfMovHasTrx(iMov)
            tgtLiveInFrm = frm2trxTotAnd(frm,iTgt);
            if ~tgtLiveInFrm
              tfInvalid(irow) = true;
              continue;
            end
          else
            assert(iTgt==1);
          end
          
          lposIFrmTgt = lposI(:,:,frm,iTgt);
          lpostagIFrmTgt = lpostagI(:,frm,iTgt);
          lposTSIFrmTgt = lposTSI(:,frm,iTgt);
          pAcc(irow,:) = lposIFrmTgt(:).'; % Shape.xy2vec(lposIFrmTgt);
          pTSAcc(irow,:) = lposTSIFrmTgt'; 
          tfoccAcc(irow,:) = lpostagIFrmTgt'; 
          
          if tfTrx && tfMovHasTrx(iMov)
            %xtrxs = cellfun(@(xx)xx(iTgt).x(frm+xx(iTgt).off),trxI);
            %ytrxs = cellfun(@(xx)xx(iTgt).y(frm+xx(iTgt).off),trxI);
            trxItgt = trxI(iTgt);
            frmabs = frm + trxItgt.off;
            xtrxs = trxItgt.x(frmabs);
            ytrxs = trxItgt.y(frmabs);
            
            pTrxAcc(irow,:) = [xtrxs(:)' ytrxs(:)']; 
            %thetas = cellfun(@(xx)xx(iTgt).theta(frm+xx(iTgt).off),trxI);
            thetas = trxItgt.theta(frmabs);
            thetaTrxAcc(irow,:) = thetas(:)'; 
            
%             as = cellfun(@(xx)xx(iTgt).a(frm+xx(iTgt).off),trxI);
%             bs = cellfun(@(xx)xx(iTgt).b(frm+xx(iTgt).off),trxI);
            as = trxItgt.a(frmabs);
            bs = trxItgt.b(frmabs);            
            aTrxAcc(irow,:) = as(:)'; 
            bTrxAcc(irow,:) = bs(:)'; 
          else
            % none; these arrays pre-initted to nan
            
%             pTrxAcc(irow,:) = nan; % singleton exp
%             thetaTrxAcc(irow,:) = nan; % singleton exp
%             aTrxAcc(irow,:) = nan; 
%             bTrxAcc(irow,:) = nan; 
          end
          nRowsComplete = nRowsComplete + 1;
        end
      end
      
      tLbl = table(pAcc,pTSAcc,tfoccAcc,pTrxAcc,thetaTrxAcc,aTrxAcc,bTrxAcc,...
        'VariableNames',{'p' 'pTS' 'tfocc' 'pTrx' 'thetaTrx' 'aTrx' 'bTrx'});
      tblMF = [tblMF tLbl];
      
      if any(tfInvalid)
        warningNoTrace('Removed %d invalid rows of MFTable.',nnz(tfInvalid));
        tblMF = tblMF(~tfInvalid,:);
      end       
    end
    
    function tblMF = lblFileGetLabels(lblfile,varargin)
      % Get all labeled rows from a lblfile
      %
      % lblfile: either char/fullpath, or struct from loaded lblfile
      
      [quiet,gt] = myparse(varargin,...
        'quiet',false,...
        'gt',false ...
        );
      
      if quiet
        wbObj = [];
      else
        wbObj = WaitBarWithCancelCmdline('Reading labels');
      end
      
      if ischar(lblfile)
        lbl = load(lblfile,'-mat');
      else
        lbl = lblfile;
      end
      if gt
        lpos = lbl.labeledposGT;
        lpostag = lbl.labeledpostagGT;
        lposts = lbl.labeledposTSGT;
      else
        lpos = lbl.labeledpos;
        lpostag = lbl.labeledpostag;
        lposts = lbl.labeledposTS;
      end
      
      tblMF = [];
      nmov = numel(lpos);
      for imov=1:nmov
        lp = lpos{imov};
        [ipts,d,frm,iTgt] = ind2sub(lp.size,lp.idx);
        tblI = table(frm,iTgt);
        tblI = unique(tblI);
        tblI.mov = MovieIndex(repmat(imov,height(tblI),1));
        
        tblMF = [tblMF; tblI]; %#ok<AGROW>
      end
      
      tblMF = tblMF(:,MFTable.FLDSID);
      
      lposfull = cellfun(@SparseLabelArray.full,lpos,'uni',0);
      lpostagfull = cellfun(@SparseLabelArray.full,lpostag,'uni',0);
      lpostsfull = cellfun(@SparseLabelArray.full,lposts,'uni',0);
      
      sMacro = lbl.projMacros;
      if gt
        mfa = lbl.movieFilesAllGT;
        tfa = lbl.trxFilesAllGT;
      else
        mfa = lbl.movieFilesAll;
        tfa = lbl.trxFilesAll;
      end
      mfafull = FSPath.fullyLocalizeStandardize(mfa,sMacro);
      tfafull = Labeler.trxFilesLocalize(tfa,mfafull);
            
      tblMF = Labeler.labelAddLabelsMFTableStc(tblMF,...
        lposfull,lpostagfull,lpostsfull,...
        'trxFilesAllFull',tfafull,...
        'trxCache',containers.Map(),...
        'wbObj',wbObj);      
    end
    
%     % Legacy meth. labelGetMFTableLabeledStc is new method but assumes
%     % .hasTrx
%     %#3DOK
%     function [I,tbl] = lblCompileContentsRaw(...
%         movieNames,lposes,lpostags,iMovs,frms,varargin)
%       % Read moviefiles with landmark labels
%       %
%       % movieNames: [NxnView] cellstr of movienames
%       % lposes: [N] cell array of labeledpos arrays [npts x 2 x nfrms x ntgts]. 
%       %   For multiview, npts=nView*NumLabelPoints.
%       % lpostags: [N] cell array of labeledpostags [npts x nfrms x ntgts]
%       % iMovs. [M] (row) indices into movieNames to read.
%       % frms. [M] cell array. frms{i} is a vector of frames to read for
%       % movie iMovs(i). frms{i} may also be:
%       %     * 'all' indicating "all frames" 
%       %     * 'lbl' indicating "all labeled frames" (currently includes partially-labeled)
%       %
%       % I: [NtrlxnView] cell vec of images
%       % tbl: [NTrl rows] labels/metadata MFTable.
%       %   MULTIVIEW NOTE: tbl.p is the 2d/projected label positions, ie
%       %   each shape has nLabelPoints*nView*2 coords, raster order is 1. pt
%       %   index, 2. view index, 3. coord index (x vs y)
%       %
%       % Optional PVs:
%       % - hWaitBar. Waitbar object
%       % - noImg. logical scalar default false. If true, all elements of I
%       % will be empty.
%       % - lposTS. [N] cell array of labeledposTS arrays [nptsxnfrms]
%       % - movieNamesID. [NxnView] Like movieNames (input arg). Use these
%       % names in tbl instead of movieNames. The point is that movieNames
%       % may be macro-replaced, platformized, etc; otoh in the MD table we
%       % might want macros unreplaced, a standard format etc.
%       % - tblMovArray. Scalar logical, defaults to false. Only relevant for
%       % multiview data. If true, use array of movies in tbl.mov. Otherwise, 
%       % use single compactified string ID.
%       
%       [hWB,noImg,lposTS,movieNamesID,tblMovArray] = myparse(varargin,...
%         'hWaitBar',[],...
%         'noImg',false,...
%         'lposTS',[],...
%         'movieNamesID',[],...
%         'tblMovArray',false);
%       assert(numel(iMovs)==numel(frms));
%       for i = 1:numel(frms)
%         val = frms{i};
%         assert(isnumeric(val) && isvector(val) || ismember(val,{'all' 'lbl'}));
%       end
%       
%       tfWB = ~isempty(hWB);
%       
%       assert(iscellstr(movieNames));
%       [N,nView] = size(movieNames);
%       assert(iscell(lposes) && iscell(lpostags));
%       assert(isequal(N,numel(lposes),numel(lpostags)));
%       tfLposTS = ~isempty(lposTS);
%       if tfLposTS
%         assert(numel(lposTS)==N);
%       end
%       for i=1:N
%         assert(size(lposes{i},1)==size(lpostags{i},1) && ...
%                size(lposes{i},3)==size(lpostags{i},2));
%         if tfLposTS
%           assert(isequal(size(lposTS{i}),size(lpostags{i})));
%         end
%       end
%       
%       if ~isempty(movieNamesID)
%         assert(iscellstr(movieNamesID));
%         szassert(movieNamesID,size(movieNames)); 
%       else
%         movieNamesID = movieNames;
%       end
%       
%       for iVw=nView:-1:1
%         mr(iVw) = MovieReader();
%       end
% 
%       I = [];
%       % Here, for multiview, mov are for the first movie in each set
%       s = struct('mov',cell(0,1),'frm',[],'p',[],'tfocc',[]);
%       
%       nMov = numel(iMovs);
%       fprintf('Reading %d movies.\n',nMov);
%       if nView>1
%         fprintf('nView=%d.\n',nView);
%       end
%       for i = 1:nMov
%         iMovSet = iMovs(i);
%         lpos = lposes{iMovSet}; % npts x 2 x nframes
%         lpostag = lpostags{iMovSet};
% 
%         [npts,d,nFrmAll] = size(lpos);
%         assert(d==2);
%         if isempty(lpos)
%           assert(isempty(lpostag));
%           lpostag = cell(npts,nFrmAll); % edge case: when lpos/lpostag are [], uninitted/degenerate case
%         end
%         szassert(lpostag,[npts nFrmAll]);
%         D = d*npts;
%         % Ordering of d is: {x1,x2,x3,...xN,y1,..yN} which for multiview is
%         % {xp1v1,xp2v1,...xpnv1,xp1v2,...xpnvk,yp1v1,...}. In other words,
%         % in decreasing raster order we have 1. pt index, 2. view index, 3.
%         % coord index (x vs y)
%         
%         for iVw=1:nView
%           movfull = movieNames{iMovSet,iVw};
%           mr(iVw).open(movfull);
%         end
%         
%         movID = MFTable.formMultiMovieID(movieNamesID(iMovSet,:));
%         
%         % find labeled/tagged frames (considering ALL frames for this
%         % movie)
%         tfLbled = arrayfun(@(x)nnz(~isnan(lpos(:,:,x)))>0,(1:nFrmAll)');
%         frmsLbled = find(tfLbled);
%         tftagged = ~cellfun(@isempty,lpostag); % [nptxnfrm]
%         ntagged = sum(tftagged,1);
%         frmsTagged = find(ntagged);
%         assert(all(ismember(frmsTagged,frmsLbled)));
% 
%         frms2Read = frms{i};
%         if strcmp(frms2Read,'all')
%           frms2Read = 1:nFrmAll;
%         elseif strcmp(frms2Read,'lbl')
%           frms2Read = frmsLbled;
%         end
%         nFrmRead = numel(frms2Read);
%         
%         ITmp = cell(nFrmRead,nView);
%         fprintf('  mov(set) %d, D=%d, reading %d frames\n',iMovSet,D,nFrmRead);
%         
%         if tfWB
%           hWB.Name = 'Reading movies';
%           wbStr = sprintf('Reading movie %s',movID);
%           waitbar(0,hWB,wbStr);
%         end
%         for iFrm = 1:nFrmRead
%           if tfWB
%             waitbar(iFrm/nFrmRead,hWB);
%           end
%           
%           f = frms2Read(iFrm);
% 
%           if noImg
%             % none; ITmp(iFrm,:) will have [] els
%           else
%             for iVw=1:nView
%               im = mr(iVw).readframe(f);
%               if size(im,3)==3 && isequal(im(:,:,1),im(:,:,2),im(:,:,3))
%                 im = rgb2gray(im);
%               end
%               ITmp{iFrm,iVw} = im;
%             end
%           end
%           
%           lblsFrmXY = lpos(:,:,f);
%           tags = lpostag(:,f);
%           
%           if tblMovArray
%             assert(false,'Unsupported codepath');
%             %s(end+1,1).mov = movieNamesID(iMovSet,:); %#ok<AGROW>
%           else
%             s(end+1,1).mov = iMovSet; %#ok<AGROW>
%           end
%           %s(end).movS = movS1;
%           s(end).frm = f;
%           s(end).p = Shape.xy2vec(lblsFrmXY);
%           s(end).tfocc = strcmp('occ',tags(:)');
%           if tfLposTS
%             lts = lposTS{iMovSet};
%             s(end).pTS = lts(:,f)';
%           end
%         end
%         
%         I = [I;ITmp]; %#ok<AGROW>
%       end
%       tbl = struct2table(s,'AsArray',true);      
%     end
        
  end
  
  %% ViewCal
  methods (Access=private)
    function viewCalCheckCalRigObj(obj,crObj)
      % Basic checks on calrig obj
      
      if ~(isequal(crObj,[]) || isa(crObj,'CalRig')&&isscalar(crObj))
        error('Labeler:viewCal','Invalid calibration object.');
      end
      nView = obj.nview;
      if nView~=crObj.nviews
        error('Labeler:viewCal',...
          'Number of views in project inconsistent with calibration object.');
      end
%     AL 20181108 very strict check, don't worry about this for now. Maybe 
%     later
%       if ~all(strcmpi(obj.viewNames(:),crObj.viewNames(:)))
%         warningNoTrace('Labeler:viewCal',...
%           'Project viewnames do not match viewnames in calibration object.');
%       end
    end
    
    function [tfAllSame,movWidths,movHeights] = viewCalCheckMovSizes(obj)
      % Check for consistency of movie sizes in current proj. Throw
      % warndlgs for each view where sizes differ.
      %
      % This considers the raw movie sizes and ignores any cropping.
      % 
      % tfAllSame: [1 nView] logical. If true, all movies in that view
      % have the same size. This includes both .movieInfoAll AND 
      % .movieInfoAllGT.
      % movWidths, movHeights: [nMovSetxnView] arrays
            
      ifo = cat(1,obj.movieInfoAll,obj.movieInfoAllGT);
      movWidths = cellfun(@(x)x.info.Width,ifo); % raw movie width
      movHeights = cellfun(@(x)x.info.Height,ifo); % raw movie height
      nrow = obj.nmovies + obj.nmoviesGT;
      nView = obj.nview;
      szassert(movWidths,[nrow nView]);
      szassert(movHeights,[nrow nView]);
      
      tfAllSame = true(1,nView);
      if nrow>0
        for iVw=1:nView
          tfAllSame(iVw) = ...
            all(movWidths(:,iVw)==movWidths(1,iVw)) && ...
            all(movHeights(:,iVw)==movHeights(1,iVw));          
        end
        if ~all(tfAllSame)
          warnstr = 'The movies in this project have varying view/image sizes. This probably doesn''t work well with calibrations. Proceed at your own risk.';
          warndlg(warnstr,'Image sizes vary','modal');
        end
      end
    end
    
%     function viewCalSetCheckViewSizes(obj,iMov,crObj,tfSet)
%       % Check/set movie image size for movie iMov on calrig object 
%       %
%       % The raw movie sizes are used here, ignoring any cropping.
%       % 
%       % iMov: movie index, applied in GT-aware fashion (eg .currMovie)
%       % crObj: scalar calrig object
%       % tfSet: if true, set the movie size on the calrig (with diagnostic
%       % printf); if false, throw warndlg if the sizes don't match
%             
%       assert(iMov>0);
%       movInfo = obj.movieInfoAllGTaware(iMov,:);
%       movWidths = cellfun(@(x)x.info.Width,movInfo); % raw movie width/height
%       movHeights = cellfun(@(x)x.info.Height,movInfo); % etc
%       vwSizes = [movWidths(:) movHeights(:)];
%       if tfSet
%         % If movie sizes differ in this project, setting of viewsizes may
%         % be hazardous. Assume warning has been thrown if necessary
%         crObj.viewSizes = vwSizes;
%         for iVw=1:obj.nview
%           fprintf(1,'Calibration obj: set [width height] = [%d %d] for view %d (%s).\n',...
%             vwSizes(iVw,1),vwSizes(iVw,2),iVw,crObj.viewNames{iVw});
%         end
%       else
%         % Check view sizes
%         if ~isequal(crObj.viewSizes,vwSizes)
%           warnstr = sprintf('View sizes in calibration object (%s) do not match movie %d (%s).',...
%             mat2str(crObj.viewSizes),iMov,mat2str(vwSizes));
%           warndlg(warnstr,'View size mismatch','non-modal');
%         end
%       end
%     end
    
  end
  methods
    
    function viewCalClear(obj)
      obj.viewCalProjWide = [];
      obj.viewCalibrationData = [];
      obj.viewCalibrationDataGT = [];
      % Currently lblCore is not cleared, change will be reflected in
      % labelCore at next movie change etc
      
      % lc = obj.lblCore;      
      % if lc.supportsCalibration
      %   warning('Labeler:viewCal','');
      % end
    end
    
    function viewCalSetProjWide(obj,crObj,varargin)
      % Set project-wide calibration object.
      %
      % .viewCalibrationData or .viewCalibrationDataGT set depending on
      % .gtIsGTMode.
      
%       tfSetViewSizes = myparse(varargin,...
%         'tfSetViewSizes',false); % If true, set viewSizes on crObj per current movieInfo
      
      if obj.nmovies==0 || obj.currMovie==0
        error('Labeler:calib',...
          'Add/select a movie first before setting the calibration object.');
      end
      
      obj.viewCalCheckCalRigObj(crObj);
      
      vcdPW = obj.viewCalProjWide;
      if ~isempty(vcdPW) && ~vcdPW
        warningNoTrace('Labeler:viewCal',...
          'Discarding movie-specific calibration data. Calibration data will apply to all movies.');
        obj.viewCalProjWide = true;
        obj.viewCalibrationData = [];
        obj.viewCalibrationDataGT = [];
      end
      
      obj.viewCalCheckMovSizes();
%       obj.viewCalSetCheckViewSizes(obj.currMovie,crObj,tfSetViewSizes);
      
      obj.viewCalProjWide = true;
      obj.viewCalibrationData = crObj;
      obj.viewCalibrationDataGT = [];

      lc = obj.lblCore;
      if lc.supportsCalibration
        lc.projectionSetCalRig(crObj);
      else
        warning('Labeler:viewCal','Current labeling mode does not utilize view calibration.');
      end
    end
    
    function viewCalSetCurrMovie(obj,crObj,varargin)
      % Set calibration object for current movie

%       tfSetViewSizes = myparse(varargin,...
%         'tfSetViewSizes',false); % If true, set viewSizes on crObj per current movieInfo
      
      nMov = obj.nmoviesGTaware;
      if nMov==0 || obj.currMovie==0
        error('Labeler:calib',...
          'Add/select a movie first before setting the calibration object.');
      end

      obj.viewCalCheckCalRigObj(crObj);

      vcdPW = obj.viewCalProjWide;
      if isempty(vcdPW)
        obj.viewCalProjWide = false;
        obj.viewCalibrationData = cell(obj.nmovies,1);
        obj.viewCalibrationDataGT = cell(obj.nmoviesGT,1);
      elseif vcdPW
        warningNoTrace('Labeler:viewCal',...
          'Discarding project-wide calibration data. Calibration data will need to be set on other movies.');
        obj.viewCalProjWide = false;
        obj.viewCalibrationData = cell(obj.nmovies,1);
        obj.viewCalibrationDataGT = cell(obj.nmoviesGT,1);
      else
        assert(iscell(obj.viewCalibrationData));
        assert(iscell(obj.viewCalibrationDataGT));
        szassert(obj.viewCalibrationData,[obj.nmovies 1]);
        szassert(obj.viewCalibrationDataGT,[obj.nmoviesGT 1]);
      end
      
%       obj.viewCalSetCheckViewSizes(obj.currMovie,crObj,tfSetViewSizes);
      PROPS = obj.gtGetSharedProps();
      obj.(PROPS.VCD){obj.currMovie} = crObj;
      
      lc = obj.lblCore;
      if lc.supportsCalibration
        lc.projectionSetCalRig(crObj);
      else
        warning('Labeler:viewCal',...
          'Current labeling mode does not utilize view calibration.');
      end
    end
    
  end
  
  methods (Static)
    function nptsLbled = labelPosNPtsLbled(lpos)
      % poor man's export of LabelPosLabeledFramesStats
      %
      % lpos: [nPt x d x nFrm x nTgt]
      % 
      % nptsLbled: [nFrm]. Number of labeled (non-nan) points for each frame
      
      [~,d,nfrm,ntgt] = size(lpos);
      assert(d==2);
      assert(ntgt==1,'One target only.');
      lposnnan = ~isnan(lpos);
      
      nptsLbled = nan(nfrm,1);
      for f = 1:nfrm
        tmp = all(lposnnan(:,:,f),2); % both x/y must be labeled for pt to be labeled
        nptsLbled(f) = sum(tmp);
      end
    end
  end
  
  methods (Access=private)
    
    function labelsUpdateNewFrame(obj,force)
      %ticinfo = tic;
      if obj.isinit
        return;
      end
      if exist('force','var')==0
        force = false;
      end
      %fprintf('labelsUpdateNewFrame 1: %f\n',toc(ticinfo)); ticinfo = tic;
      if ~isempty(obj.lblCore) && (obj.prevFrame~=obj.currFrame || force)
        obj.lblCore.newFrame(obj.prevFrame,obj.currFrame,obj.currTarget);
      end
      %fprintf('labelsUpdateNewFrame 2: %f\n',toc(ticinfo)); ticinfo = tic;
      obj.prevAxesLabelsUpdate();
      %fprintf('labelsUpdateNewFrame 3: %f\n',toc(ticinfo)); ticinfo = tic;
      obj.labels2VizUpdate('dotrkres',true);
      %fprintf('labelsUpdateNewFrame 4: %f\n',toc(ticinfo)); 
    end
    
    function labelsUpdateNewTarget(obj,prevTarget)
      if ~isempty(obj.lblCore)
        obj.lblCore.newTarget(prevTarget,obj.currTarget,obj.currFrame);
      end
      obj.prevAxesLabelsUpdate();
      obj.labels2VizUpdate('dotrkres',true,'setlbls',false,'setprimarytgt',true);
    end
    
    function labelsUpdateNewFrameAndTarget(obj,prevFrm,prevTgt)
      if ~isempty(obj.lblCore)
        obj.lblCore.newFrameAndTarget(...
          prevFrm,obj.currFrame,...
          prevTgt,obj.currTarget);
      end
      obj.prevAxesLabelsUpdate();
      obj.labels2VizUpdate('dotrkres',true,'setprimarytgt',true);
    end
        
  end
   
  %% GT mode
  methods
    function gtSetGTMode(obj,tf,varargin)
      validateattributes(tf,{'numeric' 'logical'},{'scalar'});
      
      warnChange = myparse(varargin,...
        'warnChange',false); % if true, throw a warning that GT mode is changing
      
      tf = logical(tf);
      if tf==obj.gtIsGTMode
        % none, value unchanged
      else
        if warnChange
          if tf
            warningNoTrace('Entering Ground-Truthing mode.');
          else
            warningNoTrace('Leaving Ground-Truthing mode.');
          end
        end
        obj.gtIsGTMode = tf;
        nMov = obj.nmoviesGTaware;
        if nMov==0
          obj.movieSetNoMovie();
        else
          IMOV = 1; % FUTURE: remember last/previous iMov in "other" gt mode
          obj.movieSet(IMOV);
        end
        obj.updateFrameTableComplete();
        obj.notify('gtIsGTModeChanged');
      end
    end
    function gtThrowErrIfInGTMode(obj)
      if obj.gtIsGTMode
        error('Labeler:gt','Unsupported when in GT mode.');
      end
    end
    function PROPS = gtGetSharedProps(obj)
      PROPS = Labeler.gtGetSharedPropsStc(obj.gtIsGTMode);
    end
%     function gtInitSuggestions(obj,gtSuggType,nSamp)
%       % Init/set GT suggestions using gtGenerateSuggestions
%       tblMFT = obj.gtGenerateSuggestions(gtSuggType,nSamp);
%       obj.gtSetUserSuggestions(tblMFT);
%     end
    function gtSetUserSuggestions(obj,tblMFT,varargin)
      % Set user-specified/defined GT suggestions
      %
      % tblMFT: .mov (MovieIndices), .frm, .iTgt. If [], default to all
      % labeled GT rows in proj
      
      sortcanonical = myparse(varargin,...
        'sortcanonical',false);
      
      if isequal(tblMFT,[])
        fprintf(1,'Defaulting to all labeled GT frames in project...\n');
        tblMFT = obj.labelGetMFTableLabeled('useTrain',0,'mftonly',true);
        fprintf(1,'... found %d GT rows.\n',height(tblMFT));
      end
      
      if ~istable(tblMFT) && ~all(tblfldscontains(tblMFT,MFTable.FLDSID))
        error('Specified table is not a valid Movie-Frame-Target table.');
      end
      
      tblMFT = tblMFT(:,MFTable.FLDSID);
      
      if ~isa(tblMFT.mov,'MovieIndex')
        warningNoTrace('Table .mov is numeric. Assuming positive indices into GT movie list (.movieFilesAllGT).');
        tblMFT.mov = MovieIndex(tblMFT.mov,true);
      end
      
      if isempty(tblMFT)
        % pass
      else
        [tf,tfGT] = tblMFT.mov.isConsistentSet();
        if ~(tf && tfGT)
          error('All MovieIndices in input table must reference GT movies.');
        end
        
        n0 = height(tblMFT);
        n1 = height(unique(tblMFT(:,MFTable.FLDSID)));
        if n0~=n1
          error('Input table appears to contain duplicate rows.');
        end
        
        if sortcanonical
          tblMFT2 = MFTable.sortCanonical(tblMFT);
          if ~isequal(tblMFT2,tblMFT)
            warningNoTrace('Sorting table into canonical row ordering.');
            tblMFT = tblMFT2;
          end
        else
          % UI requires sorting by movies; hopefully the movie sort leaves
          % row ordering otherwise undisturbed. This appears to be the case
          % in 2017a.
          %
          % See issue #201. User has a gtSuggestions table that is not fully
          % sorted by movie, but with a desired random row order within each
          % movie. A full/canonical sorting would be undesireable.
          tblMFT2 = sortrows(tblMFT,{'mov'},{'descend'}); % descend as gt movieindices are negative
          if ~isequal(tblMFT2,tblMFT)
            warningNoTrace('Sorting table by movie.');
            tblMFT = tblMFT2;
          end
        end
      end
      
      obj.gtSuggMFTable = tblMFT;
      obj.gtUpdateSuggMFTableLbledComplete();
      obj.gtTblRes = [];
      obj.notify('gtSuggUpdated');
      obj.notify('gtResUpdated');      
    end
    function gtUpdateSuggMFTableLbledComplete(obj,varargin)
      % update .gtUpdateSuggMFTableLbled from .gtSuggMFTable/.labeledposGT
      
      donotify = myparse(varargin,...
        'donotify',false); 
      
      tbl = obj.gtSuggMFTable;
      if isempty(tbl)
        obj.gtSuggMFTableLbled = false(0,1);
        if donotify
          obj.notify('gtSuggUpdated'); % use this event for full/general update
        end
        return;
      end
      
      lposCell = obj.labeledposGT;
      fcn = @(zm,zf,zt) (nnz(isnan(lposCell{-zm}(:,:,zf,zt)))==0);
      % a mft row is labeled if all pts are either labeled, or estocc, or
      % fullocc (lpos will be inf which is not nan)
      tfAllTgtsLbled = rowfun(fcn,tbl,...
        'InputVariables',{'mov' 'frm' 'iTgt'},...
        'OutputFormat','uni');
      szassert(tfAllTgtsLbled,[height(tbl) 1]);
      obj.gtSuggMFTableLbled = tfAllTgtsLbled;
      if donotify
        obj.notify('gtSuggUpdated'); % use this event for full/general update
      end
    end
    function gtUpdateSuggMFTableLbledIncremental(obj)
      % Assume that .labeledposGT and .gtSuggMFTableLbled differ at most in
      % currMov/currTarget/currFrame
      
      assert(obj.gtIsGTMode);
      % If not gtMode, currMovie/Frame/Target do not apply to GT
      % movies/labels. Maybe call gtUpdateSuggMFTableLbledComplete() in
      % this case.
      
      iMov = obj.currMovie;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      tblGT = obj.gtSuggMFTable;
      
      tfInTbl = tblGT.mov==(-iMov) & tblGT.frm==frm & tblGT.iTgt==iTgt;
      nRow = nnz(tfInTbl);
      if nRow>0
        assert(nRow==1);
        lposXY = obj.labeledposGT{iMov}(:,:,frm,iTgt);
        obj.gtSuggMFTableLbled(tfInTbl) = nnz(isnan(lposXY))==0;
        obj.notify('gtSuggMFTableLbledUpdated');
      end
    end
%     function tblMFT = gtGenerateSuggestions(obj,gtSuggType,nSamp)
%       assert(isa(gtSuggType,'GTSuggestionType'));
%       
%       % Start with full table (every frame), then sample
%       mfts = MFTSetEnum.AllMovAllTgtAllFrm;
%       tblMFT = mfts.getMFTable(obj);
%       tblMFT = gtSuggType.sampleMFTTable(tblMFT,nSamp);
%     end
    function [tf,idx] = gtCurrMovFrmTgtIsInGTSuggestions(obj)
      % Determine if current movie/frm/target is in gt suggestions.
      % 
      % tf: scalar logical
      % idx: if tf==true, row index into .gtSuggMFTable. If tf==false, idx
      %  is indeterminate.
      
      assert(obj.gtIsGTMode);
      mIdx = obj.currMovIdx;
      frm = obj.currFrame;
      iTgt = obj.currTarget;
      tblGT = obj.gtSuggMFTable;
      tf = mIdx==tblGT.mov & frm==tblGT.frm & iTgt==tblGT.iTgt;
      idx = find(tf);
      assert(isempty(idx) || isscalar(idx));
      tf = ~isempty(idx);
    end
    function tblMFT_SuggAndLbled = gtGetTblSuggAndLbled(obj)
      % Compile table of GT suggestions with their labels.
      % 
      % tblMFT_SuggAndLbled: Labeled GT table, in order of tblMFTSugg. To
      % be included, a row must be i) labeled for at least one pt/coord and
      % ii) in gtSuggMFTable

      
      tblMFTSugg = obj.gtSuggMFTable;
      mfts = MFTSet(MovieIndexSetVariable.AllGTMov,...
        FrameSetVariable.LabeledFrm,FrameDecimationFixed(1),...
        TargetSetVariable.AllTgts);    
      tblMFTLbld = mfts.getMFTable(obj);
      
      [tfSuggAnyLbl,loc] = tblismember(tblMFTSugg,tblMFTLbld,MFTable.FLDSID);

      % tblMFTLbld includes rows where any pt/coord is labeled;
      % obj.gtSuggMFTableLbled is only true if all pts/coords labeled 
      tfSuggFullyLbled = obj.gtSuggMFTableLbled;
      assert(all(tfSuggAnyLbl(tfSuggFullyLbled)));
      tfSuggPartiallyLbled = tfSuggAnyLbl & ~tfSuggFullyLbled;
      tfSuggUnLbled = ~tfSuggAnyLbl;
      
      nSuggUnlbled = nnz(tfSuggUnLbled);
      if nSuggUnlbled>0
        warningNoTrace('Labeler:gt',...
          '%d suggested GT frames have not been labeled.',nSuggUnlbled);
      end

      nSuggPartiallyLbled = nnz(tfSuggPartiallyLbled);
      if nSuggPartiallyLbled>0
        warningNoTrace('Labeler:gt',...
          '%d suggested GT frames have only been partially labeled.',nSuggPartiallyLbled);
      end
      
      nSuggAnyLbled = nnz(tfSuggAnyLbl);
      nTotGTLbled = height(tblMFTLbld);
      if nTotGTLbled>nSuggAnyLbled
        warningNoTrace('Labeler:gt',...
          '%d labeled GT frames were not in list of suggestions. These labels will NOT be used in assessing GT performance.',...
          nTotGTLbled-nSuggAnyLbled);
      end
      
      % Labeled GT table, in order of tblMFTSugg
      tblMFT_SuggAndLbled = tblMFTLbld(loc(tfSuggAnyLbl),:);
    end
    function tblGTres = gtComputeGTPerformance(obj,varargin)
      %
      % Front door entry point for computing gt performance
      
      [doreport,useLabels2,doui] = myparse(varargin,...
        'doreport',true, ... % if true, call .gtReport at end
        'useLabels2',false, ... % if true, use labels2 "imported preds" instead of tracking
        'doui',true ... % if true, msgbox when done
        );
      
      tObj = obj.tracker;
      if ~useLabels2 && isa(tObj,'DeepTracker')
        % Separate codepath here. DeepTrackers run in a separate async
        % process spawned by shell; trackGT in this process and then
        % remaining GT computations are done at callback time (in
        % DeepTracker.m)
        [tfsucc,msg] = tObj.trackGT();
        DIALOGTTL = 'GT Tracking';
        if tfsucc
          msg = 'Tracking of GT frames spawned. GT results will be shown when tracking is complete.';
          msgbox(msg,DIALOGTTL);
        else
          msg = sprintf('GT tracking failed: %s',msg);
          warndlg(msg,DIALOGTTL);
        end
        return;
      end

      tblMFT_SuggAndLbled = obj.gtGetTblSuggAndLbled();
      fprintf(1,'Computing GT performance with %d GT rows.\n',...
        height(tblMFT_SuggAndLbled));
        
      if useLabels2
        if ~obj.gtIsGTMode
          error('Project is not in Ground-Truthing mode.');
          % Only b/c in the next .labelGet* call we want GT mode. 
          % Pretty questionable. .labelGet* could accept GT flag
        end
        
        wbObj = WaitBarWithCancel('Compiling Imported Predictions');
        oc = onCleanup(@()delete(wbObj));
        tblTrkRes = obj.labelGetMFTableLabeled('wbObj',wbObj,...          
          'useLabels2',true,... % in GT mode, so this compiles labels2GT
          'tblMFTrestrict',tblMFT_SuggAndLbled);        
        if wbObj.isCancel
          tblGTres = [];
          warningNoTrace('Labeler property .gtTblRes not set.');
          return;
        end
        
        tblTrkRes.pTrk = tblTrkRes.p; % .p is imported positions => imported tracking
        tblTrkRes(:,'p') = [];
      else
        tObj.track(tblMFT_SuggAndLbled);
        tblTrkRes = tObj.getAllTrackResTable();
      end

      tblGTres = obj.gtComputeGTPerformanceTable(tblMFT_SuggAndLbled,tblTrkRes);
      
      if doreport
        obj.gtReport();
      end
      if doui        
        msgbox('GT results available in Labeler property ''gtTblRes''.');
      end
    end
    function tblGTres = gtComputeGTPerformanceTable(obj,tblMFT_SuggAndLbled,...
        tblTrkRes,varargin)
      % Compute GT performance 
      % 
      % tblMFT_SuggAndLbled: MFTable, no shape/label field (eg .p or
      % .pLbl). This will be added with .labelAddLabelsMFTable.
      % tblTrkRes: MFTable, predictions/tracking in field .pTrk. .pTrk is
      % in absolute coords if rois are used.
      %
      % At the moment, all rows of tblMFT_SuggAndLbled must be in tblTrkRes
      % (wrt MFTable.FLDSID). ie, there must be tracking results for all
      % suggested/labeled rows.
      %
      % All position-fields (.pTrk, .pLbl, .pTrx) in output/result are in 
      % absolute coords.
      % 
      % Assigns .gtTblRes.

      pTrkOccThresh = myparse(varargin,...
        'pTrkOccThresh',0.5 ... % threshold for predicted occlusions
        );
      
      [tf,loc] = tblismember(tblMFT_SuggAndLbled,tblTrkRes,MFTable.FLDSID);
      if ~all(tf)
        warningNoTrace('Tracking/prediction results not present for %d GT rows. Results will be computed with those rows removed.',...
          nnz(~tf));
        tblMFT_SuggAndLbled = tblMFT_SuggAndLbled(tf,:);
        loc = loc(tf);
      end      
      tblTrkRes = tblTrkRes(loc,:);
      
      tblMFT_SuggAndLbled = obj.labelAddLabelsMFTable(tblMFT_SuggAndLbled);
      pTrk = tblTrkRes.pTrk; % absolute coords
      pLbl = tblMFT_SuggAndLbled.p; % absolute coords
      nrow = size(pTrk,1);
      npts = obj.nLabelPoints;
      szassert(pTrk,[nrow 2*npts]);
      szassert(pLbl,[nrow 2*npts]);
      
      % L2 err matrix: [nrow x npt]
      pTrk = reshape(pTrk,[nrow npts 2]);
      pLbl = reshape(pLbl,[nrow npts 2]);
      err = sqrt(sum((pTrk-pLbl).^2,3));
      
      tflblinf = any(isinf(pLbl),3); % [nrow x npts] fully-occ indicator mat; lbls currently coded as inf
      err(tflblinf) = nan; % treat fully-occ err as nan here
      muerr = nanmean(err,2); % and ignore in meanL2err
      
      % ctab for occlusion pred
%       % this is not adding value yet
%       tfTrkHasOcc = isfield(tblTrkRes,'pTrkocc');
%       if tfTrkHasOcc        
%         pTrkocctf = tblTrkRes.pTrkocc>=pTrkOccThresh;
%         szassert(pTrkocctf,size(tflblinf));
%       end        
      
      tblTmp = tblMFT_SuggAndLbled(:,{'p' 'pTS' 'tfocc' 'pTrx'});
      tblTmp.Properties.VariableNames = {'pLbl' 'pLblTS' 'tfoccLbl' 'pTrx'};
      if tblfldscontains(tblTrkRes,'pTrx')
        assert(isequal(tblTrkRes.pTrx,tblTmp.pTrx),'Mismatch in .pTrx fields.');
        tblTrkRes(:,'pTrx') = [];
      end
      tblGTres = [tblTrkRes tblTmp table(err,muerr,'VariableNames',{'L2err' 'meanL2err'})];
      
      obj.gtTblRes = tblGTres;
      obj.notify('gtResUpdated');
    end
    function h = gtReport(obj,varargin)
      t = obj.gtTblRes;

      nmontage = myparse(varargin,...
        'nmontage',height(t));      
      
      t.meanOverPtsL2err = mean(t.L2err,2);
      % KB 20181022: Changed colors to match sets instead of points
      clrs =  obj.LabelPointColors;
      nclrs = size(clrs,1);
      npts = size(t.L2err,2);
      assert(npts==obj.nLabelPoints);
      if nclrs~=npts
        warningNoTrace('Labeler:gt',...
          'Number of colors do not match number of points.');
      end
      
      % Err by landmark
      h = figure('Name','GT err by landmark');
      ax = axes;
      boxplot(t.L2err,'colors',clrs,'boxstyle','filled');
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Landmark/point',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,'GT err by landmark',args{:});
      ax.YGrid = 'on';
      
      % AvErrAcrossPts by movie
      h(end+1,1) = figurecascaded(h(end),'Name','Mean GT err by movie');
      ax = axes;
      [iMovAbs,gt] = t.mov.get;
      assert(all(gt));
      grp = categorical(iMovAbs);
      grplbls = arrayfun(@(z1,z2)sprintf('mov%s (n=%d)',z1{1},z2),...
        categories(grp),countcats(grp),'uni',0);
      boxplot(t.meanOverPtsL2err,grp,'colors',clrs,'boxstyle','filled',...
        'labels',grplbls);
      args = {'fontweight' 'bold' 'interpreter' 'none'};
      xlabel(ax,'Movie',args{:});
      ylabel(ax,'L2 err (px)',args{:});
      title(ax,'Mean (over landmarks) GT err by movie',args{:});
      ax.YGrid = 'on';
      
      % Mean err by movie, pt
      h(end+1,1) = figurecascaded(h(end),'Name','Mean GT err by movie, landmark');
      ax = axes;
      tblStats = grpstats(t(:,{'mov' 'L2err'}),{'mov'});
      tblStats.mov = tblStats.mov.get;
      tblStats = sortrows(tblStats,{'mov'});
      movUnCnt = tblStats.GroupCount; % [nmovx1]
      meanL2Err = tblStats.mean_L2err; % [nmovxnpt]
      nmovUn = size(movUnCnt,1);
      szassert(meanL2Err,[nmovUn npts]);
      meanL2Err(:,end+1) = nan; % pad for pcolor
      meanL2Err(end+1,:) = nan;       
      hPC = pcolor(meanL2Err);
      hPC.LineStyle = 'none';
      colorbar;
      xlabel(ax,'Landmark/point',args{:});
      ylabel(ax,'Movie',args{:});
      xticklbl = arrayfun(@num2str,1:npts,'uni',0);
      yticklbl = arrayfun(@(x)sprintf('mov%d (n=%d)',x,movUnCnt(x)),1:nmovUn,'uni',0);
      set(ax,'YTick',0.5+(1:nmovUn),'YTickLabel',yticklbl);
      set(ax,'XTick',0.5+(1:npts),'XTickLabel',xticklbl);
      axis(ax,'ij');
      title(ax,'Mean GT err (px) by movie, landmark',args{:});
      
      nmontage = min(nmontage,height(t));
      obj.trackLabelMontage(t,'meanOverPtsL2err','hPlot',h,'nplot',nmontage);
    end    
    function gtNextUnlabeledUI(obj)
      % Like pressing "Next Unlabeled" in GTManager.
      if obj.gtIsGTMode
        gtMgr = obj.gdata.GTMgr;
        gd = guidata(gtMgr);
        pb = gd.pbNextUnlabeled;
        cbk = pb.Callback;
        cbk(pb,[]);
      else
        warningNoTrace('Not in GT mode.');
      end
    end
    function gtShowGTManager(obj)
      hGTMgr = obj.gdata.GTMgr;
      hGTMgr.Visible = 'on';
      figure(hGTMgr);
    end
    function [iMov,iMovGT] = gtCommonMoviesRegGT(obj)
      % Find movies common to both regular and GT lists
      %
      % For multiview projs, movienames must match across all views
      % 
      % iMov: vector of positive ints, reg movie(set) indices
      % iMovGT: " gt movie(set) indices

      [iMov,iMovGT] = Labeler.identifyCommonMovSets(...
        obj.movieFilesAllFull,obj.movieFilesAllGTFull);
    end
    function fname = getDefaultFilenameExportGTResults(obj)
      gtstr = 'gtresults';
      if ~isempty(obj.projectfile)
        rawname = ['$projdir/$projfile_' gtstr '.mat'];
      elseif ~isempty(obj.projname)
        rawname = ['$projdir/$projname_' gtstr '.mat'];
      else
        rawname = ['$projdir/' gtstr '.mat'];
      end
      sMacro = obj.baseTrkFileMacros();
      fname = FSPath.macroReplace(rawname,sMacro);
    end
    function tbl = gtLabeledFrameSummary(obj)
      % return/print summary of gt movies with number of labels
      
      imov = obj.gtSuggMFTable.mov.abs();
      imovun = unique(imov);
      tflbld = obj.gtSuggMFTableLbled;

      imovuncnt = arrayfun(@(x)nnz(x==imov),imovun);
      imovunlbledcnt = arrayfun(@(x)nnz(x==imov & tflbld),imovun);      
      tbl = table(imovun,imovunlbledcnt,imovuncnt,...
        'VariableNames',{'GT Movie Index' 'Labeled Frames' 'Total GT Frames'});
    end
  end
  methods (Static)
    function PROPS = gtGetSharedPropsStc(gt)
      PROPS = Labeler.PROPS_GTSHARED;
      if gt
        PROPS = PROPS.gt;
      else
        PROPS = PROPS.reg;
      end
    end
  end
  
  %% Susp 
  methods
    
    function suspInit(obj)
      obj.suspScore = cell(size(obj.labeledpos));
      obj.suspSelectedMFT = MFTable.emptySusp;
      obj.suspComputeFcn = [];
    end
    
    function suspSetComputeFcn(obj,fcn)
      assert(isa(fcn,'function_handle'));      
      obj.suspInit();
      obj.suspComputeFcn = fcn;
    end
    
    function tfsucc = suspCompute(obj)
      % Populate .suspScore, .suspSelectedMFT by calling .suspComputeFcn
      
      fcn = obj.suspComputeFcn;
      if isempty(fcn)
        error('Labeler:susp','No suspiciousness function has been set.');
      end
      [suspscore,tblsusp,diagstr] = fcn(obj);
      if isempty(suspscore)
        % cancel/fail
        warningNoTrace('Labeler:susp','No suspicious scores computed.');
        tfsucc = false;
        return;
      end
      
      obj.suspVerifyScore(suspscore);
      if ~isempty(tblsusp)
        if ~istable(tblsusp) && all(ismember(MFTable.FLDSSUSP,...
                                  tblsusp.Properties.VariableNames'))
          error('Labeler:susp',...
            'Invalid ''tblsusp'' output from suspicisouness computation.');
        end
      end
      
      obj.suspScore = suspscore;
      obj.suspSelectedMFT = tblsusp;
      obj.suspDiag = diagstr;
      
      tfsucc = true;
    end
    
    function suspComputeUI(obj)
      tfsucc = obj.suspCompute();
      if ~tfsucc
        return;
      end
      figtitle = sprintf('Suspicious frames: %s',obj.suspDiag);
      hF = figure('Name',figtitle);
      tbl = obj.suspSelectedMFT;
      tblFlds = tbl.Properties.VariableNames;
      nt = NavigationTable(hF,[0 0 1 1],@(i,rowdata)obj.suspCbkTblNaved(i),...
        'ColumnName',tblFlds);
      nt.setData(tbl);
%       nt.navOnSingleClick = true;
      hF.UserData = nt;
%       kph = SuspKeyPressHandler(nt);
%       setappdata(hF,'keyPressHandler',kph);

      obj.addDepHandle(hF);
    end
    
    function suspCbkTblNaved(obj,i)
      % i: row index into .suspSelectedMFT;
      tbl = obj.suspSelectedMFT;
      nrow = height(tbl);
      if i<1 || i>nrow
        error('Labeler:susp','Row ''%d'' out of bounds.',i);
      end
      mftrow = tbl(i,:);
      if obj.currMovie~=mftrow.mov
        obj.movieSet(mftrow.mov);
      end
      obj.setFrameAndTarget(mftrow.frm,mftrow.iTgt);
    end
  end
  
  methods (Hidden)
    
    function suspVerifyScore(obj,suspscore)
      nmov = obj.nmovies;
      if ~(iscell(suspscore) && numel(suspscore)==nmov)
        error('Labeler:susp',...
          'Invalid ''suspscore'' output from suspicisouness computation.');
      end
      lpos = obj.labeledpos;
      for imov=1:nmov
        [~,~,nfrm,ntgt] = size(lpos{imov});
        if ~isequal(size(suspscore{imov}),[nfrm ntgt])
          error('Labeler:susp',...
            'Invalid ''suspscore'' output from suspicisouness computation.');
        end
      end
    end
  end

    
        
%     function setSuspScore(obj,ss)
%       assert(~obj.isMultiView);
%       
%       if isequal(ss,[])
%         % none; this is ok
%       else
%         nMov = obj.nmovies;
%         nTgt = obj.nTargets;
%         assert(iscell(ss) && isvector(ss) && numel(ss)==nMov);
%         for iMov = 1:nMov
%           ifo = obj.movieInfoAll{iMov,1}; 
%           assert(isequal(size(ss{iMov}),[ifo.nframes nTgt]),...
%             'Size mismatch for score for movie %d.',iMov);
%         end
%       end
%       
%       obj.suspScore = ss;
%     end
    
%     function updateCurrSusp(obj)
%       % Update .currSusp from .suspScore, currMovie, .currFrm, .currTarget
%       
%       tfDoSusp = ~isempty(obj.suspScore) && ...
%                   obj.hasMovie && ...
%                   obj.currMovie>0 && ...
%                   obj.currFrame>0;
%       if tfDoSusp
%         ss = obj.suspScore{obj.currMovie};
%         obj.currSusp = ss(obj.currFrame,obj.currTarget);       
%       else
%         obj.currSusp = [];
%       end
%       if ~isequal(obj.currSusp,[])
%         obj.currImHud.updateSusp(obj.currSusp);
%       end
%     end
  
%% PreProc
%  
% "Preprocessing" represents transformations taken on raw movie/image data
% in preparation for tracking: eg histeq, cropping, nbormasking, bgsubbing.
% The hope is that these transformations can normalize/clean up data to
% increase overall tracking performance. One imagines that many preproc
% steps apply equally well to any particular tracking algorithm; then there
% may be algorithm-specific preproc steps as well.
% 
% Preproc computations are governed by a set of parameters. Within the UI
% these are available under the Track->Configure Tracking Parameters menu.
% The user is also able to turn on/off preprocessing steps as they desire.
% 
% Conceptually Preproc is a linear pipeline accepting raw movie data as
% the input and producing images ready for consumption by the tracker. 
% Ideally the PreProc pipeline is the SOLE SOURCE of input data (besides
% parameters) for trackers.
% 
% APT caches preprocessed data for two reasons. It can be convenient from a
% performance standpoint, eg if one is iteratively retraining and gauging
% performance on a test set of frames. Another reason is that by caching
% data, users can conveniently browse training data, diagnostic metadata,
% etc.
%
% For CPR, the PP cache is in-memory. For DL trackers, the PP cache is on
% disk as the PP data/images ultimately need to be written to disk anyway
% for tracking.
% 
% When a preprocessing parameter is mutated, the pp cache must be cleared.
% We define an invariant so that any live data in the PP cache must have
% been generated with the current PP parameters. However, there is a
% question about whether a trained tracker should also be cleared, if a PP
% parameter is altered.
% 
% The conservative route is to always clear any trained tracker if any PP
% parameter is mutated. However, there may be cases where this is
% undesirable. Viewed as a black box, the tracker accepts images and
% produces annotations; the fact that it was trained on images produced in
% a certain way doesn't guarantee that a user might not want to apply it to
% images produced in a (possibly slightly) different way. For instance,
% after a tracker has already been trained, a background image for a movie
% might be incrementally improved. Using the old/trained tracker with data 
% PPed using the new background image with bgsub might be perfectly
% reasonable. There may be other interesting cases as well, eg train with
% no nbor mask, track with nbor mask, where "more noise" during training is
% desireable to improve generalization.
% 
% With preprocessing clearly separated from tracking, it should be fairly
% easy to enable these more complex scenarios in the future if desired.
% 
% Parameter mutation Summary
% - PreProc parameter mutated -> PP cache cleared; trained tracker cleared (maybe in future, something fancier)
% - Tracking parameter mutated -> PP cache untouched; trained tracker cleared
% 
% NOTE: During a retrain(), a snapshot of preprocParams should be taken and
% saved with the trained tracker. That way it is recorded/known precisely 
% how that tracker was generated.

  methods
    
    function preProcInit(obj)
      %obj.preProcParams = [];
      obj.preProcH0 = [];
      obj.preProcInitData();
      obj.ppdbInit();
      obj.preProcSaveData = false;
      obj.movieFilesAllHistEqLUT = cell(obj.nmovies,obj.nview);
      obj.movieFilesAllGTHistEqLUT = cell(obj.nmoviesGT,obj.nview);
    end
    
    function preProcInitData(obj)
      % Initialize .preProcData*
      
      I = cell(0,1);
      tblP = MFTable.emptyTable(MFTable.FLDSCORE);
      obj.preProcData = CPRData(I,tblP);
      obj.preProcDataTS = now;
    end
    
    function ppdbInit(obj)
      if isempty(obj.ppdb)
        obj.ppdb = PreProcDB();
      end
      obj.ppdb.init();
    end
    
%     function tfPPprmsChanged = preProcSetParams(obj,ppPrms) % THROWS
%       % ppPrms: OLD-style preproc params
%       
%       assert(isstruct(ppPrms));
% 
%       if ppPrms.histeq 
%         if ppPrms.BackSub.Use
%           error('Histogram Equalization and Background Subtraction cannot both be enabled.');
%         end
%         if ppPrms.NeighborMask.Use
%           error('Histogram Equalization and Neighbor Masking cannot both be enabled.');
%         end
%       end
%       
%       ppPrms0 = obj.preProcParams;
%       tfPPprmsChanged = ~isequaln(ppPrms0,ppPrms);
%       if tfPPprmsChanged
%         warningNoTrace('Preprocessing parameters altered; data cache cleared.');
%         obj.preProcInitData();
%         obj.ppdbInit(); % AL20190123: currently only ppPrms.TargetCrop affect ppdb
%         
%         bgPrms = ppPrms.BackSub;
%         mrs = obj.movieReader;
%         for i=1:numel(mrs)
%           mrs(i).open(mrs(i).filename,'bgType',bgPrms.BGType,...
%             'bgReadFcn',bgPrms.BGReadFcn); 
%           % mrs(i) should already be faithful to .forceGrayscale, 
%           % .movieInvert, cropInfo
%         end
%       end
%       obj.preProcParams = ppPrms;
%     end
    
    function preProcNonstandardParamChanged(obj)
      % Normally, preProcSetParams() handles changes to preproc-related
      % parameters. If a preproc parameter changes, then the output
      % variable tfPPprmsChanged is set to true, and the caller 
      % clears/updates the tracker as necessary.
      %
      % There are two nonstandard pre-preprocessing "parameters" that are 
      % set outside of preProcSetParams: .movieInvert, and
      % .movieFilesAll*CropInfo. If these properties are mutated, any
      % preprocessing data, trained tracker, and tracking results must also
      % be cleared.
      
      obj.preProcInitData();
      obj.ppdbInit();
      for i=1:numel(obj.trackersAll)
        if obj.trackersAll{i}.getHasTrained()
          warningNoTrace('Trained tracker(s) and tracking results cleared.');
          break;
        end
      end
      obj.trackInitAllTrackers();
    end
    
    function tblP = preProcCropLabelsToRoiIfNec(obj,tblP,varargin)
      % Add .roi column to table if appropriate/nec
      %
      % Preconds: One of these must hold
      %   - proifld and pabsfld are both not fields/cols
      %     - if roi is present as a field, its existing value is checked/confirmed
      %   - None of {roi, proifld, pabsfld} are present
      %
      % PostConditions:
      % If hasTrx, modify tblP as follows:
      %   - add .roi
      %   - add .pRoi (proifld), p relative to ROI
      %   - set .pAbs (pabsfld) to the original/absolute .p (pfld)
      %   - set .p (pfld) to be .pRoi (proifld)
      % If cropProjHasCrops, same as hasTrx.
      % Otherwise:
      %   - no .roi
      %   - set .pAbs (pabsfld) to be .p (pfld)
      
      [prmpp,doRemoveOOB,pfld,pabsfld,proifld] = myparse(varargin,...
        'preProcParams',[],...
        'doRemoveOOB',true,...
        'pfld','p',...  % see desc above
        'pabsfld','pAbs',... % etc
        'proifld','pRoi'... % 
        );
      isPreProcParams = ~isempty(prmpp);
      
      if obj.hasTrx || obj.cropProjHasCrops
        % at the end of this branch, all fields .roi, proifld, pabsfld must
        % be added
        
        tf2pflds = tblfldscontains(tblP,{proifld pabsfld});
        tfroi = tblfldscontains(tblP,'roi');
        %assert(all(tf2pflds) || ~any(tf2pflds));
        if ~any(tf2pflds)
          if tfroi
            % save existing .roi fld and confirm it matches the one that
            % will be added below
            roi0 = tblP.roi;
            tblP(:,'roi') = [];
          end
          if obj.hasTrx
            if isPreProcParams,
              roiRadius = prmpp.TargetCrop.Radius;
            else
              roiRadius = obj.preProcParams.TargetCrop.Radius;
            end
            tblP = obj.labelMFTableAddROITrx(tblP,roiRadius,...
              'rmOOB',doRemoveOOB,...
              'pfld',pfld,'proifld',proifld);
          else
            tblP = obj.labelMFTableAddROICrop(tblP,...
              'rmOOB',doRemoveOOB,...
              'pfld',pfld,'proifld',proifld);
          end
          if tfroi
            assert(isequaln(roi0,tblP.roi));
          end
          tblP.(pabsfld) = tblP.(pfld);
          tblP.(pfld) = tblP.(proifld);
        else % one of the two fields exists
          assert(all(tf2pflds) && tfroi);
        end
      else
        if tblfldscontains(tblP,pabsfld) % AL20190207 add this now some downstream clients want it
          assert(isequaln(tblP.(pfld),tblP.(pabsfld)));
        else
          tblP.(pabsfld) = tblP.(pfld);
        end
        % none; tblP.p is .pAbs. No .roi field.
      end
    end
    
    function tblP = preProcGetMFTableLbled(obj,varargin)
      % labelGetMFTableLabeled + preProcCropLabelsToRoiIfNec
      %
      % Get MFTable for all movies/labeledframes. Exclude partially-labeled 
      % frames.
      %
      % tblP: MFTable of labeled frames. Precise cols may vary. However:
      % - MFTable.FLDSFULL are guaranteed where .p is:
      %   * The absolute position for single-target trackers
      %   * The position relative to .roi for multi-target trackers
      % - .roi is guaranteed when .hasTrx or .cropProjHasCropInfo

      [wbObj,tblMFTrestrict,gtModeOK,prmpp,doRemoveOOB,...
        treatInfPosAsOcc] = myparse(varargin,...
        'wbObj',[], ... % optional WaitBarWithCancel. If cancel:
                    ... % 1. obj const 
                    ... % 2. tblP indeterminate
        'tblMFTrestrict',[],... % see labelGetMFTableLabeld
        'gtModeOK',false,... % by default, this meth should not be called in GT mode
        'preProcParams',[],...
        'doRemoveOOB',true,...
        'treatInfPosAsOcc',false ... % if true, treat inf labels as 
                                 ... % 'fully occluded'; if false, remove 
                                 ... % any rows with inf labels
        ); 
      tfWB = ~isempty(wbObj);
      if ~isempty(tblMFTrestrict)
        tblfldsassert(tblMFTrestrict,MFTable.FLDSID);
      end
      
      if obj.gtIsGTMode && ~gtModeOK
        error('Unsupported in GT mode.');
      end
      
      tblP = obj.labelGetMFTableLabeled('wbObj',wbObj,...
        'tblMFTrestrict',tblMFTrestrict);
      if tfWB && wbObj.isCancel
        % tblP indeterminate, return it anyway
        return;
      end
      
      % In tblP we can have:
      % * regular labels: .p is non-nan & non-inf; .tfocc is false
      % * estimated-occluded labels: .p is non-nan & non-inf; .tfocc is true
      % * fully-occ labels: .p is inf, .tfocc is false

      % For now we don't accept partially-labeled rows
      tfnanrow = any(isnan(tblP.p),2);
      nnanrow = nnz(tfnanrow);
      if nnanrow>0
        warningNoTrace('Labeler:nanData',...
          'Not including %d partially-labeled rows.',nnanrow);
      end
      tblP = tblP(~tfnanrow,:);
        
      % Deal with full-occ rows in tblP. Do this here as otherwise the call 
      % to preProcCropLabelsToRoiIfNec will consider inf labels as "OOB"
      if treatInfPosAsOcc
        tfinf = isinf(tblP.p);        
        pAbsIsFld = any(strcmp(tblP.Properties.VariableNames,'pAbs'));
        if pAbsIsFld % probably .pAbs is put in below by preProcCropLabelsToRoiIfNec
          assert(isequal(tfinf,isinf(tblP.pAbs)));
        end
        tfinf2 = reshape(tfinf,height(tblP),[],2);
        assert(isequal(tfinf2(:,:,1),tfinf2(:,:,2))); % both x/y coords should be inf for fully-occ
        nfulloccpts = nnz(tfinf2(:,:,1));
        if nfulloccpts>0
          warningNoTrace('Utilizing %d fully-occluded landmarks.',nfulloccpts);
        end
      
        tblP.p(tfinf) = nan;
        if pAbsIsFld
          tblP.pAbs(tfinf) = nan;
        end
        tblP.tfocc(tfinf2(:,:,1)) = true;
      else
        tfinf = any(isinf(tblP.p),2);
        ninf = nnz(tfinf);
        if ninf>0
          warningNoTrace('Labeler:infData',...
            'Not including %d rows with fully-occluded labels.',ninf);
        end
        tblP = tblP(~tfinf,:);
      end
      
      tblP = obj.preProcCropLabelsToRoiIfNec(tblP,'preProcParams',prmpp,...
        'doRemoveOOB',doRemoveOOB);
    end
    
    % Hist Eq Notes
    %
    % The Labeler has the ability/responsibility to compute typical image
    % histograms representative of all movies in the current project, for
    % each view. See movieEstimateImHist. 
    %
    % .preProcH0 stores this typical hgram for use in preprocessing. 
    % Conceptually, .preProcH0 is like .preProcParams in that it is a 
    % parameter (vector) governing how data is preprocessed. Instead of 
    % being user-set like other parameters, .preProcH0 is updated/learned 
    % from the project movies.
    %
    % .preProcH0 is set at retrain- (fulltraining-) time. It is not updated
    % during tracking, or during incremental trains. If users are adding
    % movies/labels, they should periodically retrain fully, to refresh 
    % .preProcH0 relative to the movies.
    %
    % Note that .preProcData has its own .H0 for historical reasons. This
    % should always coincide with .preProcH0, except possibly in edge cases
    % where one or the other is empty.
    % 
    % SUMMARY MAIN OPERATIONS
    % retrain: this updates .preProcH0 and also forces .preProcData.H0
    % to match; .preProcData is initially cleared if necessary.
    % inctrain: .preProcH0 is untouched. .preProcData.H0 continues to match 
    % .preProcH0.
    % track: same as inctrain.
    
    function preProcUpdateH0IfNec(obj)
      % Update obj.preProcH0
      % Update .movieFilesAllHistEqLUT, .movieFilesAllGTHistEqLUT 

      % AL20180910: currently using frame-by-frame CLAHE
      USECLAHE = true;
      if USECLAHE
        obj.preProcH0 = [];
        return;
      end
      
      
      ppPrms = obj.preProcParams;
      if ppPrms.histeq
        nFrmSampH0 = ppPrms.histeqH0NumFrames;
        s = struct();
        [s.hgram,s.hgraminfo] = obj.movieEstimateImHist(...
          'nFrmPerMov',nFrmSampH0,'debugViz',false);
        
        data = obj.preProcData;
        if data.N>0 && ~isempty(obj.preProcH0) && ~isequal(data.H0,obj.preProcH0.hgram)
          assert(false,'.preProcData.H0 differs from .preProcH0');
        end
        if ~isequal(data.H0,s.hgram)
          obj.preProcInitData();
          % Note, currently we do not clear trained tracker/tracking
          % results, see above. This is caller's responsibility. Atm all
          % callers do a retrain or equivalent
        end
        obj.preProcH0 = s;
        
        wbObj = WaitBarWithCancel('Computing Histogram Matching LUTs',...
          'cancelDisabled',true);
        oc = onCleanup(@()delete(wbObj));
        obj.movieEstimateHistEqLUTs('nFrmPerMov',nFrmSampH0,...
          'wbObj',wbObj,'docheck',true);
      else
        assert(isempty(obj.preProcData.H0));
        
        % For now we don't force .preProcH0, .movieFilesAll*HistEq* to be
        % empty. User can compute them, then turn off HistEq, and the state
        % remains
        if false
          assert(isempty(obj.preProcH0));
          tf = cellfun(@isempty,obj.movieFilesAllHistEqLUT);
          tfGT = cellfun(@isempty,obj.movieFilesAllGTHistEqLUT);
          assert(all(tf(:)));
          assert(all(tfGT(:)));
        end
      end
    end
    
    function [data,dataIdx,tblP,tblPReadFailed,tfReadFailed] = ...
        preProcDataFetch(obj,tblP,varargin)
      % dataUpdate, then retrieve
      %
      % Input args: See PreProcDataUpdate
      %
      % data: CPRData handle, equal to obj.preProcData
      % dataIdx. data.I(dataIdx,:) gives the rows corresponding to tblP
      %   (out); order preserved
      % tblP (out): subset of tblP (input), rows for failed reads removed
      % tblPReadFailed: subset of tblP (input) where reads failed
      % tfReadFailed: indicator vec into tblP (input) for failed reads.
      %   tblP (out) is guaranteed to correspond to tblP (in) with
      %   tfReadFailed rows removed. (unless early/degenerate/empty return)
      
      % See preProcDataUpdateRaw re 'preProcParams' opt arg. When supplied,
      % .preProcData is not updated.
      
      [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
        'wbObj',[],... % WaitBarWithCancel. If cancel: obj unchanged, data and dataIdx are [].
        'updateRowsMustMatch',false, ... % See preProcDataUpdateRaw
        'preProcParams',[]...
        );
      isPreProcParamsIn = ~isempty(prmpp);
      tfWB = ~isempty(wbObj);
      
      [tblPReadFailed,data] = obj.preProcDataUpdate(tblP,'wbObj',wbObj,...
        'updateRowsMustMatch',updateRowsMustMatch,'preProcParams',prmpp);
      if tfWB && wbObj.isCancel
        data = [];
        dataIdx = [];
        tblP = [];
        tblPReadFailed = [];
        tfReadFailed = [];
        return;
      end
      
      if ~isPreProcParamsIn,
        data = obj.preProcData;
      end
      tfReadFailed = tblismember(tblP,tblPReadFailed,MFTable.FLDSID);
      tblP(tfReadFailed,:) = [];
      [tf,dataIdx] = tblismember(tblP,data.MD,MFTable.FLDSID);
      assert(all(tf));
    end
    
    function [tblPReadFailed,dataNew] = preProcDataUpdate(obj,tblP,varargin)
      % Update .preProcData to include tblP
      %
      % tblP:
      %   - MFTable.FLDSCORE: required.
      %   - .roi: optional, USED WHEN PRESENT. (prob needs to be either
      %   consistently there or not-there for a given obj or initData()
      %   "session"
      %   IMPORTANT: if .roi is present, .p (labels) are expected to be 
      %   relative to the roi.
      %   - .pTS: optional (if present, deleted)
      
      [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
        'wbObj',[],... % WaitBarWithCancel. If cancel, obj unchanged.
        'updateRowsMustMatch',false, ... % See preProcDataUpdateRaw
        'preProcParams',[]...
        );
      
      if any(strcmp('pTS',tblP.Properties.VariableNames))
        % AL20170530: Not sure why we do this
        tblP(:,'pTS') = [];
      end
      isPreProcParamsIn = ~isempty(prmpp);
      if isPreProcParamsIn,
        tblPnew = tblP;
        tblPupdate = tblP([],:);
      else
        [tblPnew,tblPupdate] = obj.preProcData.tblPDiff(tblP);
      end
      [tblPReadFailed,dataNew] = obj.preProcDataUpdateRaw(tblPnew,tblPupdate,...
        'wbObj',wbObj,'updateRowsMustMatch',updateRowsMustMatch,...
        'preProcParams',prmpp);
    end
    
    function [tblPReadFailed,dataNew] = preProcDataUpdateRaw(obj,...
        tblPnew,tblPupdate,varargin)
      % Incremental data update
      %
      % * Rows appended and pGT/tfocc updated; but other information
      % untouched
      % * histeq (if enabled) uses .preProcH0. See "Hist Eq Notes" below.
      % .preProcH0 is NOT updated here.
      %
      % QUESTION: why is pTS not updated?
      %
      % tblPNew: new rows. MFTable.FLDSCORE are required fields. .roi may 
      %   be present and if so WILL BE USED to grab images and included in 
      %   data/MD. Other fields are ignored.
      %   IMPORTANT: if .roi is present, .p (labels) are expected to be 
      %   relative to the roi.
      %
      % tblPupdate: updated rows (rows with updated pGT/tfocc).
      %   MFTable.FLDSCORE fields are required. Only .pGT and .tfocc are 
      %   otherwise used. Other fields ignored, INCLUDING eg .roi and 
      %   .nNborMask. Ie, you cannot currently update the roi of a row in 
      %   the cache (whose image has already been fetched)
      %
      %   
      % tblPReadFailed: table of failed-to-read rows. Currently subset of
      %   tblPnew. If non-empty, then .preProcData was not updated with 
      %   these rows as requested.
      %
      % Updates .preProcData, .preProcDataTS
      
      % NOTE: when the preProcParams opt arg is [] (isPreProcParamsIn is 
      % false), this is maybe a separate method, def distinct behavior. 
      % When isPreProcParamsIn is true, .preProcData is not updated, etc.
      
      dataNew = [];
      
      [wbObj,updateRowsMustMatch,prmpp] = myparse(varargin,...
        'wbObj',[], ... % Optional WaitBarWithCancel obj. If cancel, obj unchanged.
        'updateRowsMustMatch',false, ... % if true, assert/check that tblPupdate matches current cache
        'preProcParams',[]...
        );
      tfWB = ~isempty(wbObj);
      
      FLDSREQUIRED = MFTable.FLDSCORE;
      FLDSALLOWED = [MFTable.FLDSCORE {'roi' 'nNborMask'}];
      tblfldscontainsassert(tblPnew,FLDSREQUIRED);
      tblfldscontainsassert(tblPupdate,FLDSREQUIRED);
      
      tblPReadFailed = tblPnew([],:);
      
      isPreProcParamsIn = ~isempty(prmpp);
      if ~isPreProcParamsIn,
        prmpp = obj.preProcParams;
        if isempty(prmpp)
          error('Please specify tracking parameters.');
        end
        dataCurr = obj.preProcData;
      end
      
      USECLAHE = true;

      if prmpp.histeq
        if ~USECLAHE && isPreProcParamsIn,
          assert(dataCurr.N==0 || isequal(dataCurr.H0,obj.preProcH0.hgram));
        end
        assert(~prmpp.BackSub.Use,...
          'Histogram Equalization and Background Subtraction cannot both be enabled.');
        assert(~prmpp.NeighborMask.Use,...
          'Histogram Equalization and Neighbor Masking cannot both be enabled.');
      end
      if ~isempty(prmpp.channelsFcn)
        assert(obj.nview==1,...
          'Channels preprocessing currently unsupported for multiview tracking.');
      end
      
      %%% NEW ROWS read images + PP. Append to dataCurr. %%%
      FLDSID = MFTable.FLDSID;
      assert(isPreProcParamsIn||~any(tblismember(tblPnew,dataCurr.MD,FLDSID)));
      
      tblPNewConcrete = obj.mftTableConcretizeMov(tblPnew);
      nNew = height(tblPnew);
      if nNew>0
        fprintf(1,'Adding %d new rows to data...\n',nNew);

        [I,nNborMask,didread] = CPRData.getFrames(tblPNewConcrete,...
          'wbObj',wbObj,...
          'forceGrayscale',obj.movieForceGrayscale,...
          'preload',obj.movieReadPreLoadMovies,...
          'movieInvert',obj.movieInvert,...
          'roiPadVal',prmpp.TargetCrop.PadBkgd,...
          'doBGsub',prmpp.BackSub.Use,...
          'bgReadFcn',prmpp.BackSub.BGReadFcn,...
          'bgType',prmpp.BackSub.BGType,...
          'maskNeighbors',prmpp.NeighborMask.Use,...
          'maskNeighborsMeth',prmpp.NeighborMask.SegmentMethod,...
          'maskNeighborsEmpPDF',obj.fgEmpiricalPDF,...
          'fgThresh',prmpp.NeighborMask.FGThresh,...
          'trxCache',obj.trxCache);
        if tfWB && wbObj.isCancel
          % obj unchanged
          return;
        end
        % Include only FLDSALLOWED in metadata to keep CPRData md
        % consistent (so can be appended)
        
        didreadallviews = all(didread,2);
        tblPReadFailed = tblPnew(~didreadallviews,:);
        tblPnew(~didreadallviews,:) = [];
        I(~didreadallviews,:) = [];
        nNborMask(~didreadallviews,:) = [];
        
        % AL: a little worried if all reads fail -- might get a harderr
        
        tfColsAllowed = ismember(tblPnew.Properties.VariableNames,...
          FLDSALLOWED);
        tblPnewMD = tblPnew(:,tfColsAllowed);
        tblPnewMD = [tblPnewMD table(nNborMask)];
        
        if prmpp.histeq
          if USECLAHE
            if tfWB
              wbObj.startPeriod('Performing CLAHE','shownumden',true,...
                'denominator',numel(I));
            end
            for i=1:numel(I)
              if tfWB
                wbObj.updateFracWithNumDen(i);
              end
              I{i} = adapthisteq(I{i});
            end
            if tfWB
              wbObj.endPeriod();
            end
            dataNew = CPRData(I,tblPnewMD);            
          else
            J = obj.movieHistEqApplyLUTs(I,tblPnewMD.mov,'wbObj',wbObj); 
            dataNew = CPRData(J,tblPnewMD);
            dataNew.H0 = obj.preProcH0.hgram;

            if ~isPreProcParamsIn && dataCurr.N==0
              dataCurr.H0 = dataNew.H0;
              % these need to match for append()
            end
          end
        else
          dataNew = CPRData(I,tblPnewMD);
        end
                
        if ~isempty(prmpp.channelsFcn)
          feval(prmpp.channelsFcn,dataNew);
          assert(~isempty(dataNew.IppInfo),...
            'Preprocessing channelsFcn did not set .IppInfo.');
          if ~isPreProcParamsIn && isempty(dataCurr.IppInfo)
            assert(dataCurr.N==0,'Ippinfo can be empty only for empty/new data.');
            dataCurr.IppInfo = dataNew.IppInfo;
          end
        end
        
        if ~isPreProcParamsIn,
          dataCurr.append(dataNew);
        end
      end
      
      %%% EXISTING ROWS -- just update pGT and tfocc. Existing images are
      %%% OK and already histeq'ed correctly
      nUpdate = size(tblPupdate,1);
      if ~isPreProcParamsIn && nUpdate>0 % AL 20160413 Shouldn't need to special-case, MATLAB 
                   % table indexing API may not be polished
        [tf,loc] = tblismember(tblPupdate,dataCurr.MD,FLDSID);
        assert(all(tf));
        if updateRowsMustMatch
          assert(isequal(dataCurr.MD{loc,'tfocc'},tblPupdate.tfocc),...
            'Unexpected discrepancy in preproc data cache: .tfocc field');
          if tblfldscontains(tblPupdate,'roi')
            assert(isequal(dataCurr.MD{loc,'roi'},tblPupdate.roi),...
              'Unexpected discrepancy in preproc data cache: .roi field');
          end
          if tblfldscontains(tblPupdate,'nNborMask')
            assert(isequal(dataCurr.MD{loc,'nNborMask'},tblPupdate.nNborMask),...
              'Unexpected discrepancy in preproc data cache: .nNborMask field');
          end
          assert(isequaln(dataCurr.pGT(loc,:),tblPupdate.p),...
            'Unexpected discrepancy in preproc data cache: .p field');
        else
          fprintf(1,'Updating labels for %d rows...\n',nUpdate);
          dataCurr.MD{loc,'tfocc'} = tblPupdate.tfocc; % AL 20160413 throws if nUpdate==0
          dataCurr.pGT(loc,:) = tblPupdate.p;
          % Check .roi, .nNborMask?
        end
      end
      
      if ~isPreProcParamsIn,
        if nUpdate>0 || nNew>0 % AL: if all reads fail, nNew>0 but no new rows were actually read
          assert(obj.preProcData==dataCurr); % handles; not sure why this is asserted in this branch specifically
          obj.preProcDataTS = now;
        else
          warningNoTrace('Nothing to update in data.');
        end
      end
    end
   
  end
  

  %% Tracker
  methods    
    
    function [tObj,iTrk] = trackGetTracker(obj,algoName)
      % Find a particular tracker
      %
      % algoName: char, to match LabelTracker.algorithmName
      %
      % tObj: either [], or scalar tracking object 
      % iTrk: either 0, or index into .trackersAll
      for iTrk=1:numel(obj.trackersAll)
        if strcmp(obj.trackersAll{iTrk}.algorithmName,algoName)
          tObj = obj.trackersAll{iTrk};
          return;
        end
      end
      tObj = [];
      iTrk = 0;
    end
  
    function trackSetCurrentTracker(obj,iTrk)
      validateattributes(iTrk,{'numeric'},...
        {'nonnegative' 'integer' '<=' numel(obj.trackersAll)});
      
      tAll = obj.trackersAll;
      iTrk0 = obj.currTracker;
      if iTrk0>0
        tAll{iTrk0}.setHideViz(true);
      end
      obj.currTracker = iTrk;
      if iTrk>0
        tAll{iTrk}.setHideViz(false);
      end
      obj.labelingInit();
    end
    
    function sPrm = setTrackNFramesParams(obj,sPrm)
      obj.trackNFramesSmall = sPrm.ROOT.Track.NFramesSmall;
      obj.trackNFramesLarge = sPrm.ROOT.Track.NFramesLarge;
      obj.trackNFramesNear = sPrm.ROOT.Track.NFramesNeighborhood;
      sPrm.ROOT.Track = rmfield(sPrm.ROOT.Track,{'NFramesSmall','NFramesLarge','NFramesNeighborhood'});
    end
    
    function trackSetParams(obj,sPrm,varargin)
      % Set all parameters:
      %  - preproc
      %  - cpr
      %  - common dl
      %  - specific dl
      % 
      % sPrm: scalar struct containing *NEW*-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack
      
      [setall] = myparse(varargin,...
        'all',false... % if true, sPrm can contain 'extra parameters' like fliplandmarks. no callsites currently
        ); 
      sPrm = APTParameters.enforceConsistency(sPrm);

      [tfOK,msgs] = APTParameters.checkParams(sPrm);
      if ~tfOK,
        error('%s. ',msgs{:});
      end
      
      sPrm0 = obj.trackParams;
      tfPPprmsChanged = ...
        xor(isempty(sPrm0),isempty(sPrm)) || ...
        ~APTParameters.isEqualPreProcParams(sPrm0,sPrm);
      sPrm = obj.setTrackNFramesParams(sPrm);      
      if setall,
        sPrm = obj.setExtraParams(sPrm);
      end
      
      obj.trackParams = sPrm;
      
      if tfPPprmsChanged
        warningNoTrace('Preprocessing parameters altered; data cache cleared.');
        obj.preProcInitData();
        obj.ppdbInit(); % AL20190123: currently only ppPrms.TargetCrop affect ppdb
        
        bgPrms = sPrm.ROOT.ImageProcessing.BackSub;
        mrs = obj.movieReader;
        for i=1:numel(mrs)
          mrs(i).open(mrs(i).filename,'bgType',bgPrms.BGType,...
            'bgReadFcn',bgPrms.BGReadFcn);
          % mrs(i) should already be faithful to .forceGrayscale,
          % .movieInvert, cropInfo
        end
      end
      
      % KB 20190214: this will all happen at training time now -- 
      
%       tcprObj = obj.trackGetTracker('cpr');      
%       assert(~isempty(tcprObj));
%         
%       % Future TODO: right now this is hardcoded, eg "DeepTrack" doesn't
%       % match 'poseTF', and lots of special-case code.
%       % Ideally/theoretically the params/trackerObjs would just line up and
%       % setting would just be a simple loop or similar
%       
%       sPrmDT = sPrm.ROOT.DeepTrack;
%       sPrmPPandCPR = sPrm;
%       sPrmPPandCPR.ROOT = rmfield(sPrmPPandCPR.ROOT,'DeepTrack'); 
%       
%       % NOTE: this line already sets some props, despite possible throws
%       % later
%       [sPrmPPandCPRold,obj.trackNFramesSmall,obj.trackNFramesLarge,...
%         obj.trackNFramesNear] = CPRParam.new2old(sPrmPPandCPR,obj.nPhysPoints,obj.nview);
%       
%       ppPrms = sPrmPPandCPRold.PreProc;
%       sPrmCPRold = rmfield(sPrmPPandCPRold,'PreProc');
% 
%       % THROWS. Some state already mutated. Should be OK for now, its a
%       % partial set but if/when the user tries to set parameters again the
%       % changes should be reflected.
%       tfPPprmsChanged = obj.preProcSetParams(ppPrms); % THROWS
%       
%       tcprObj.setParamContentsSmart(sPrmCPRold,tfPPprmsChanged);
%       
%       tDTs = obj.trackersAll(2:end);
%       dlNetTypesPretty = cellfun(@(x)x.trnNetType.prettyString,tDTs,'uni',0);
%       sPrmDTcommon = rmfield(sPrmDT,dlNetTypesPretty);
%       tfDTcommonChanged = obj.trackSetDLParams(sPrmDTcommon);
%       tfDTcommonOrPPChanged = tfDTcommonChanged || tfPPprmsChanged;
%       for i=1:numel(tDTs)
%         tObj = tDTs{i};
%         netType = dlNetTypesPretty{i};
%         sPrmDTnet = sPrmDT.(netType);
%         tObj.setParamContentsSmart(sPrmDTnet,tfDTcommonOrPPChanged);
%       end

    end
    
    function [sPrmDT,sPrmCPRold,ppPrms,trackNFramesSmall,trackNFramesLarge,...
        trackNFramesNear] = convertNew2OldParams(obj,sPrm) % obj CONST
      % Conversion routine
      % 
      % sPrm: scalar struct containing *NEW*-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack
              
      sPrm = APTParameters.enforceConsistency(sPrm);
      
      sPrmDT = sPrm.ROOT.DeepTrack;
      sPrmPPandCPR = sPrm;
      sPrmPPandCPR.ROOT = rmfield(sPrmPPandCPR.ROOT,'DeepTrack'); 
      
      [sPrmPPandCPRold,trackNFramesSmall,trackNFramesLarge,...
        trackNFramesNear] = CPRParam.new2old(sPrmPPandCPR,obj.nPhysPoints,obj.nview);
      
      ppPrms = sPrmPPandCPRold.PreProc;
      sPrmCPRold = rmfield(sPrmPPandCPRold,'PreProc');
    end
    
    function sPrm = trackGetParams(obj,varargin)
      % Get all parameters:
      %  - preproc
      %  - cpr
      %  - common dl
      %  - specific dl
      %
      % sPrm: scalar struct containing NEW-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack (if applicable)
      % Top-level fields .Track, .CPR, .DeepTrack may be missing if they
      % don't exist yet.
      
      % Future TODO: As in trackSetParams, currently this is hardcoded when
      % it ideally would just be a generic loop
      [getall] = myparse(varargin,'all',false);
      
      sPrm = obj.trackParams;
      sPrm.ROOT.Track.NFramesSmall = obj.trackNFramesSmall;
      sPrm.ROOT.Track.NFramesLarge = obj.trackNFramesLarge;
      sPrm.ROOT.Track.NFramesNeighborhood = obj.trackNFramesNear;
      if getall,
        sPrm = obj.addExtraParams(sPrm);
      end
      
    end
    
    function be = trackGetDLBackend(obj)
      be = obj.trackDLBackEnd;
    end
    
    function trackSetDLBackend(obj,be)
      assert(isa(be,'DLBackEndClass'));
      
      switch be.type
        case DLBackEnd.AWS
          % special-case this to avoid running repeat AWS commands
          
          aws = be.awsec2;
          if isempty(aws),            
            be.awsec2 = AWSec2([],...
              'SetStatusFun',[], ... % @(varargin) obj.SetStatus(varargin{:}),...
              'ClearStatusFun',[]); % ... %@(varargin) obj.ClearStatus(varargin{:}));
          else
            % This codepath occurs on eg projLoad
            be.awsec2.SetStatusFun = []; % @(varargin) obj.SetStatus(varargin{:});
            be.awsec2.ClearStatusFun = []; % @(varargin) obj.ClearStatus(varargin{:});
          end
%           [tfexist,tfrunning] = aws.inspectInstance();
%           if tfexist
%             % AWS auto-shutdown alarm 20190213
%             % The only official way to set the APT backend is here. We add
%             % a metricalarm here to auto-shutdown the EC2 instance should
%             % it become idle.
%             %
%             % - We use use a single/unique alarm name (see AWSec2). I think
%             % this an AWS account can only have one alarm at a time, so
%             % adding it here removes it from somewhere else if it is
%             % somewhere else.
%             % - If an account uses multiple instances, some will be
%             % unprotected for now. We expect the typical use case to be a
%             % single instance at a time.
%             % - Currently we never remove the alarm, so it just hangs
%             % around configured for the last instance where it was added. I
%             % don't get the impression that this hurts or that CloudWatch
%             % is expensive etc. Note in particular, the CloudWatch alarm
%             % lifecycle is independent of the EC2 lifecycle. CloudWatch
%             % alarms specify an instance only eg via the 'Dimensions'.
%             % - The alarm(s) is clearly visible on the EC2 dash. I think it
%             % should be ok for now.
%             aws.configureAlarm;
%           end
%           
%           if isempty(aws) || ~tfexist
%             warningNoTrace('AWS backend is not configured. You will need to configure an instance before training or tracking.');
%           elseif ~tfrunning
%             warningNoTrace('AWS backend instance is not running. You will need to start instance before training or tracking.');
%           end
          
        otherwise
          [tf,reason] = be.getReadyTrainTrack();
          if ~tf
            warningNoTrace('Backend is not ready to train: %s',reason);
          end
      end
      
      obj.trackDLBackEnd = be;
    end
    
    function trackTrain(obj)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      tObj.train();
    end
        
    function trackRetrain(obj,varargin)
      [tblMFTtrn,retrainArgs,dontUpdateH0] = myparse(varargin,...
        'tblMFTtrn',[],... % (opt) table on which to train (cols MFTable.FLDSID only). defaults to all of obj.preProcGetMFTableLbled
        'retrainArgs',{},... % (opt) args to pass to tracker.retrain()
        'dontUpdateH0',false...
        );
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~obj.hasMovie
        error('Labeler:track','No movie.');
      end
      
      if ~isempty(tblMFTtrn)
        assert(strcmp(tObj.algorithmName,'cpr'));
        % assert this as we do not fetch tblMFTp to treatInfPosAsOcc
        tblMFTp = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFTtrn);
        retrainArgs = [retrainArgs(:)' {'tblPTrn' tblMFTp}];
      end           
      
	  % KB 20190121 moved this to within retrain, since we don't clear tracking results immediately for background deep learning
      % tObj.clearTrackingResults();
      if ~dontUpdateH0
        obj.preProcUpdateH0IfNec();
      end
      tObj.retrain(retrainArgs{:});
    end
    
    function [bgTrnIsRunning] = trackBGTrnIsRunning(obj)
      
      bgTrnIsRunning = false(1,numel(obj.trackersAll));
      for i = 1:numel(obj.trackersAll),
        if isprop(obj.trackersAll{i},'bgTrnIsRunning'),
          bgTrnIsRunning(i) = obj.trackersAll{i}.bgTrnIsRunning;
        end
          
      end
      
    end
    
    function [tfCanTrain,reason] = trackCanTrain(obj,varargin)
      
      tfCanTrain = false;
      % is tracker and movie
      if isempty(obj.tracker),
        reason = 'The tracker has not been set.';
        return;
      end
      if ~obj.hasMovie,
        reason = 'There must be at least one movie in the project.';
        return;
      end
      
      if isempty(obj.trackParams)
        reason = 'Tracking parameters have not been set.';
        return;
      end
      
      % parameters set, whatever other checks tracker wants to do
      [tfCanTrain,reason] = obj.tracker.canTrain();
      if ~tfCanTrain,
        return;
      end
      
      % are labels
      [tblMFTtrn] = myparse(varargin,...
        'tblMFTtrn',[]... % (opt) table on which to train (cols MFTable.FLDSID only). defaults to all of obj.preProcGetMFTableLbled
        );
      
      % allow 'treatInfPosAsOcc' to default to false in these calls; we are
      % just checking number of labeled rows
      warnst = warning('off','Labeler:infData'); % ignore "Not including n rows with fully-occ labels"
      oc = onCleanup(@()warning(warnst));
      if ~isempty(tblMFTtrn)
        tblMFTp = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFTtrn);
      else
        tblMFTp = obj.preProcGetMFTableLbled();
      end
    
      nlabels = size(tblMFTp,1);
      
      if nlabels < 2,
        tfCanTrain = false;
        reason = 'There must be at least two labeled frames in the project.';
        return;
      end
      
      tfCanTrain = true;      
      
    end
    
    function [tfCanTrack,reason] = trackCanTrack(obj,tblMFT)
      tfCanTrack = false;
      if isempty(obj.tracker),
        reason = 'The tracker has not been set.';
        return;
      end
      
      [tfCanTrack,reason] = obj.tracker.canTrack();
      
      if ~tfCanTrack,
        return;
      end
      
      [tfCanTrack,reason] = PostProcess.canPostProcess(obj,tblMFT);

      
    end
    
    function tfCanTrack = trackAllCanTrack(obj)
      
      tfCanTrack = false(1,numel(obj.trackersAll));
      for i = 1:numel(obj.trackersAll),
        tfCanTrack(i) = obj.trackersAll{i}.canTrack;
      end
      
    end
    
    function track(obj,mftset,varargin)
      % mftset: an MFTSet or table tblMFT
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end

      if isa(mftset,'table'),
        tblMFT = mftset;
      else
        assert(isa(mftset,'MFTSet'));
        tblMFT = mftset.getMFTable(obj);
      end
      
      tObj.track(tblMFT,varargin{:});
      
      % For template mode to see new tracking results
      obj.labelsUpdateNewFrame(true);
      
      %fprintf('Tracking complete at %s.\n',datestr(now));
    end
    
    function trackTbl(obj,tblMFT,varargin)
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      tObj.track(tblMFT,varargin{:});
      % For template mode to see new tracking results
      obj.labelsUpdateNewFrame(true);
      
      fprintf('Tracking complete at %s.\n',datestr(now));
    end
    
    function trackInitAllTrackers(obj)
      cellfun(@(x)x.init(),obj.trackersAll);
    end
    
    function [tfsucc,tblPCache,s] = ...
        trackCreateDeepTrackerStrippedLbl(obj,varargin)
      % For use with DeepTrackers. Create stripped lbl based on
      % .currTracker
      %
      % tfsucc: false if user canceled etc.
      % tblPCache: table of data cached in stripped lbl (eg training data, 
      %   or gt data)
      % s: scalar struct, stripped lbl struct
      
      [wbObj,ppdata,sPrmAll,shuffleRows] = myparse(varargin,...
        'wbObj',[],...
        'ppdata',[],...
        'sPrmAll',[],...
        'shuffleRows',true ...
        );
      tfWB = ~isempty(wbObj);
      
      if ~obj.hasMovie
        % for NumChans see below
        error('Please select/open a movie.');
      end
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('There is no current tracker selected.');
      end
      
      isGT = obj.gtIsGTMode;
      if isGT
        descstr = 'gt';
      else
        descstr = 'training';
      end
      
      %
      % Determine the training set
      % 
      if isempty(ppdata),
        treatInfPosAsOcc = ...
          isa(tObj,'DeepTracker') && tObj.trnNetType.doesOccPred;
        tblPCache = obj.preProcGetMFTableLbled(...
          'wbObj',wbObj,...
          'gtModeOK',isGT,...
          'treatInfPosAsOcc',treatInfPosAsOcc ...
          );
        if tfWB && wbObj.isCancel
          tfsucc = false;
          tblPCache = [];
          s = [];
          return;
        end
        
        if isempty(tblPCache)
          error('No %s data available.',descstr);
        end
        
        if obj.hasTrx
          tblfldscontainsassert(tblPCache,[MFTable.FLDSCOREROI {'thetaTrx'}]);
        elseif obj.cropProjHasCrops
          tblfldscontainsassert(tblPCache,[MFTable.FLDSCOREROI]);
        else
          tblfldscontainsassert(tblPCache,MFTable.FLDSCORE);
        end
        
        [tblAddReadFailed,tfAU,locAU] = obj.ppdb.addAndUpdate(tblPCache,obj,...
          'wbObj',wbObj);
        if tfWB && wbObj.isCancel
          tfsucc = false;
          tblPCache = [];
          s = [];
          return;
        end
        nMissedReads = height(tblAddReadFailed);
        if nMissedReads>0
          warningNoTrace('Removing %d %s rows, failed to read images.\n',...
            nMissedReads,descstr);
        end
        
        assert(all(locAU(~tfAU)==0));
        
        ppdbICache = locAU(tfAU); % row indices into obj.ppdb.dat for our training set
        tblPCache = obj.ppdb.dat.MD(ppdbICache,:);
        
        fprintf(1,'%s with %d rows.\n',descstr,numel(ppdbICache));
        fprintf(1,'%s data summary:\n',descstr);
        obj.ppdb.dat.summarize('mov',ppdbICache);        
      else
        % training set provided; note it may or may not include fully-occ
        % labels etc.
        tblPCache = ppdata.MD;
        ppdbICache = (1:ppdata.N)';
      end
      
      % 
      % Create the stripped lbl struct
      % 

      if isempty(ppdata),
        s = obj.projGetSaveStruct('forceIncDataCache',true,...
          'macroreplace',true,'massageCropProps',true);
      else
        s = obj.projGetSaveStruct('forceExcDataCache',true,...
          'macroreplace',true,'massageCropProps',true);
      end
      s.projectFile = obj.projectfile;

      nchan = arrayfun(@(x)x.getreadnchan,obj.movieReader);
      nchan = unique(nchan);
      if ~isscalar(nchan)
        error('Number of channels differs across views.');
      end
      s.cfg.NumChans = nchan; % see below, we change this again
      
%       if nchan>1
%         warningNoTrace('Images have %d channels. Typically grayscale images are preferred; select View>Convert to grayscale.',nchan);
%       end
      
      % AL: moved above
      if ~isempty(ppdata)
%         ppdbICache = true(ppdata.N,1);
      else
        % De-objectize .ppdb.dat (CPRData)
        ppdata = s.ppdb.dat;
      end
      
      fprintf(1,'Stripped lbl preproc data cache: exporting %d/%d %s rows.\n',...
        numel(ppdbICache),ppdata.N,descstr);
      
      ppdataI = ppdata.I(ppdbICache,:);
      ppdataP = ppdata.pGT(ppdbICache,:);
      ppdataMD = ppdata.MD(ppdbICache,:);
      
      ppdataMD.mov = int32(ppdataMD.mov); % MovieIndex
      ppMDflds = tblflds(ppdataMD);
      s.preProcData_I = ppdataI;
      s.preProcData_P = ppdataP;
      for f=ppMDflds(:)',f=f{1}; %#ok<FXSET>
        sfld = ['preProcData_MD_' f];
        s.(sfld) = ppdataMD.(f);
      end
      if isfield(s,'ppdb'),
        s = rmfield(s,'ppdb');
      end
      if isfield(s,'preProcData'),
        s = rmfield(s,'preProcData');
      end
      
      % 20201120 randomize training rows
      if shuffleRows
        fldsPP = fieldnames(s);
        fldsPP = fldsPP(startsWith(fldsPP,'preProcData_'));
        nTrn = size(ppdataI,1);
        prand = obj.projDeterministicRandFcn(@()randperm(nTrn));
        fprintf(1,'Shuffling training rows. Your RNG seed is: %d\n',obj.projRngSeed);
        for f=fldsPP(:)',f=f{1}; %#ok<FXSET>
          v = s.(f);
          assert(ndims(v)==2 && size(v,1)==nTrn); %#ok<ISMAT>
          s.(f) = v(prand,:);
        end
      else
        prand = [];
      end
      s.preProcData_prand = prand;
      
      s.trackerClass = {'__UNUSED__' 'DeepTracker'};
      
      %
      % Final Massage
      % 
     
      
      tdata = s.trackerData{s.currTracker};
      
      if ~isempty(sPrmAll),
        tdata.sPrmAll = sPrmAll;
      end
      
      tdata.sPrmAll = obj.addExtraParams(tdata.sPrmAll);
      
%       if tdata.trnNetType==DLNetType.openpose
%         % put skeleton => affinity_graph for now        
%         skel = obj.skeletonEdges;
%         assert(~isempty(skel));
%         nedge = size(skel,1);
%         skelstr = arrayfun(@(x)sprintf('%d %d',skel(x,1),skel(x,2)),1:nedge,'uni',0);
%         skelstr = String.cellstr2CommaSepList(skelstr);
%         tdata.sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph = skelstr;
%       else
%         tdata.sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph = '';
%       end
%       % add landmark matches
%       matches = obj.flipLandmarkMatches;
%       nedge = size(matches,1);
%       matchstr = arrayfun(@(x)sprintf('%d %d',matches(x,1),matches(x,2)),1:nedge,'uni',0);
%       matchstr = String.cellstr2CommaSepList(matchstr);
%       tdata.sPrmAll.ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches = matchstr;
%       %tdata.sPrmAll.ROOT.DeepTrack.
      
      tdata.trnNetTypeString = char(tdata.trnNetType);
      
      tftrx = obj.hasTrx;      
      if tftrx
        
        % KB 20190212: ignore sizex and sizey, these will be removed
        %roirad = s.trackParams.ROOT.ImageProcessing.MultiTarget.TargetCrop.Radius;
        %tdata.sPrm.sizex = 2*roirad+1;
        %tdata.sPrm.sizey = 2*roirad+1;

%         dlszx = tdata.sPrm.ImageProcessing.sizex;
%         dlszy = tdata.sPrm.ImageProcessing.sizey;
%         szroi = 2*roirad+1;
%         if dlszx~=szroi
%           warningNoTrace('Target ROI Radius is %d while DeepTrack sizeX is %d. Setting sizeX to %d to match ROI Radius.',roirad,dlszx,szroi);
%           tdata.sPrm.sizex = szroi;
%         end
%         if dlszy~=szroi
%           warningNoTrace('Target ROI Radius is %d while DeepTrack sizeY is %d. Setting sizeY to %d to match ROI Radius.',roirad,dlszy,szroi);
%           tdata.sPrm.sizey = szroi;
%         end
        
      end
      s.trackerData = {[] tdata};
      s.nLabels = ppdata.N;
      
      % check with Mayank, thought we wanted number of "underlying" chans
      % but DL is erring when pp data is grayscale but NumChans is 3
      s.cfg.NumChans = size(s.preProcData_I{1},3);
      
      tfsucc = true;
    end
    
    % See also Lbl.m for addnl stripped lbl meths
    
    function sPrmAll = addExtraParams(obj,sPrmAll)
            
      skel = obj.skeletonEdges;
      if ~isempty(skel),
        nedge = size(skel,1);
        skelstr = arrayfun(@(x)sprintf('%d %d',skel(x,1),skel(x,2)),1:nedge,'uni',0);
        skelstr = String.cellstr2CommaSepList(skelstr);
        sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph = skelstr;
      else
        sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph = '';
      end
      % add landmark matches
      matches = obj.flipLandmarkMatches;
      nedge = size(matches,1);
      matchstr = arrayfun(@(x)sprintf('%d %d',matches(x,1),matches(x,2)),1:nedge,'uni',0);
      matchstr = String.cellstr2CommaSepList(matchstr);
      sPrmAll.ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches = matchstr;

    end
    
    function sPrmAll = setExtraParams(obj,sPrmAll)
      % AL 20200409 sets .skeletonEdges and .setFliplandmarkMatches from 
      % sPrmAll fields. no callsites currently
      
      if structisfield(sPrmAll,'ROOT.DeepTrack.OpenPose.affinity_graph'),
        skelstr = sPrmAll.ROOT.DeepTrack.OpenPose.affinity_graph;
        nedge = numel(skelstr);
        skel = nan(nedge,2);
        for i = 1:nedge,
          skel(i,:) = sscanf(skelstr{i},'%d %d');
        end
        obj.skeletonEdges = skel;
        sPrmAll.ROOT.DeepTrack.OpenPose = rmfield(sPrmAll.ROOT.DeepTrack.OpenPose,'affinity_graph');
        if isempty(fieldnames(sPrmAll.ROOT.DeepTrack.OpenPose)),
          sPrmAll.ROOT.DeepTrack.OpenPose = '';
        end
      end

      % add landmark matches
      if structisfield(sPrmAll,'ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches'),
        matchstr = sPrmAll.ROOT.DeepTrack.DataAugmentation.flipLandmarkMatches;
        matchstr = strsplit(matchstr,',');
        nedge = numel(matchstr);
        matches = nan(nedge,2);
        for i = 1:nedge,
          matches(i,:) = sscanf(matchstr{i},'%d %d');
        end
        obj.setFlipLandmarkMatches(matches);
        sPrmAll.ROOT.DeepTrack.DataAugmentation = rmfield(sPrmAll.ROOT.DeepTrack.DataAugmentation,'flipLandmarkMatches');
        if isempty(fieldnames(sPrmAll.ROOT.DeepTrack.DataAugmentation)),
          sPrmAll.ROOT.DeepTrack.DataAugmentation = '';
        end
      end
    end
    
    function trackAndExport(obj,mftset,varargin)
      % Track one movie at a time, exporting results to .trk files and 
      % clearing data in between
      %
      % mftset: scalar MFTSet

      startTime = tic;
      
      [trackArgs,rawtrkname] = myparse(varargin,...
        'trackArgs',{},...
        'rawtrkname',[]...
        );
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end
      if ~strcmp(tObj.algorithmName,'cpr')
        % DeepTrackers track in bg and already export to trk
        error('Only CPR tracking supported.');
      end

      tblMFT = mftset.getMFTable(obj);
      
      iMovsUn = unique(tblMFT.mov);      
      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovsUn,[],...
        rawtrkname,{},'noUI',true);
      if ~tfok
        return;
      end
      
      nMov = numel(iMovsUn);
      nVw = obj.nview;
      szassert(trkfiles,[nMov nVw]);
      if obj.isMultiView
        moviestr = 'movieset';
      else
        moviestr = 'movie';
      end
      fprintf('Preprocessing time in trackAndExport: %f\n',toc(startTime)); startTime = tic;
      for i=1:nMov
        iMov = iMovsUn(i);
        fprintf('Tracking %s %d (%d/%d) -> %s\n',moviestr,double(iMov),i,...
          nMov,trkfiles{i});
        
        tfMov = tblMFT.mov==iMov;
        tblMFTmov = tblMFT(tfMov,:);
        tObj.track(tblMFTmov,trackArgs{:});
        fprintf('Tracking time: %f\n',toc(startTime)); startTime = tic;
        trkFile = tObj.getTrackingResults(iMov);
        szassert(trkFile,[1 nVw]);
        for iVw=1:nVw
          trkFile{iVw}.save(trkfiles{i,iVw});
          fprintf('...saved: %s\n',trkfiles{i,iVw});
          fprintf('Save time: %f\n',toc(startTime)); startTime = tic;
        end
        tObj.clearTrackingResults();
        fprintf('Time to clear tracking results: %f\n',toc(startTime)); startTime = tic;
        obj.preProcInitData();
        obj.ppdbInit(); % putting this here just b/c the above line, quite possibly unnec
        fprintf('Time to reinitialize data: %f\n',toc(startTime)); startTime = tic;
      end
    end
    
    function trackExportResults(obj,iMovs,varargin)
      % Export tracking results to trk files.
      %
      % iMovs: [nMov] vector of movie(set)s whose tracking should be
      % exported. iMovs are indexed into .movieFilesAllGTAware
      %
      % If a movie has no current tracking results, a warning is thrown and
      % no trkfile is created.
      
      [trkfiles,rawtrkname] = myparse(varargin,...
        'trkfiles',[],... % [nMov nView] cellstr, fullpaths to trkfilenames to export to
        'rawtrkname',[]... % string, basename to apply over iMovs to generate trkfiles
        );
            
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:track','No tracker set.');
      end

      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovs,trkfiles,...
        rawtrkname,{});
      if ~tfok
        return;
      end
      
      movfiles = obj.movieFilesAllFullGTaware(iMovs,:);
      gt = obj.gtIsGTMode;
      mIdx = MovieIndex(iMovs,gt);
      [trkFileObjs,tfHasRes] = tObj.getTrackingResults(mIdx);
      nMov = numel(iMovs);
      nVw = obj.nview;
      szassert(trkFileObjs,[nMov nVw]);
      szassert(trkfiles,[nMov nVw]);
      for iMv=1:nMov
        if tfHasRes(iMv)
          for iVw=1:nVw
            trkFileObjs{iMv,iVw}.save(trkfiles{iMv,iVw});
            fprintf('Saved %s.\n',trkfiles{iMv,iVw});
          end
        else
          if obj.isMultiView
            moviestr = 'movieset';
          else
            moviestr = 'movie';
          end
          warningNoTrace('Labeler:noRes','No current tracking results for %s %s.',...
            moviestr,MFTable.formMultiMovieID(movfiles(iMv,:)));
        end
      end
    end
        
    function trackCrossValidate(obj,varargin)
      % Run k-fold crossvalidation. Results stored in .xvResults
      
      [kFold,initData,wbObj,tblMFgt,tblMFgtIsFinal,partTst,dontInitH0] = ...
        myparse(varargin,...
        'kfold',3,... % number of folds
        'initData',false,... % OBSOLETE, you would never want this. if true, call .initData() between folds to minimize mem usage
        'wbObj',[],... % (opt) WaitBarWithCancel
        'tblMFgt',[],... % (opt), MFTable of data to consider. Defaults to all labeled rows. tblMFgt should only contain fields .mov, .frm, .iTgt. labels, rois, etc will be assembled from proj
        'tblMFgtIsFinal',false,... % a bit silly, for APT developers only. Set to true if your tblMFgt is in final form.
        'partTst',[],... % (opt) pre-defined training splits. If supplied, partTst must be a [height(tblMFgt) x kfold] logical. tblMFgt should be supplied. true values indicate test rows, false values indicate training rows.
        'dontInitH0',true...
      );        
      
      tfWB = ~isempty(wbObj);
      tfTblMFgt = ~isempty(tblMFgt);      
      tfPart = ~isempty(partTst);
      
      if obj.gtIsGTMode
        error('Unsupported in GT mode.');
      end
      
      if ~tfTblMFgt
        % CPR required below; allow 'treatInfPosAsOcc' to default to false
        tblMFgt = obj.preProcGetMFTableLbled();
      elseif ~tblMFgtIsFinal        
        tblMFgt0 = tblMFgt; % legacy checks below
        % CPR required below; allow 'treatInfPosAsOcc' to default to false
        tblMFgt = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFgt);
        % Legacy checks/assert can remove at some pt
        assert(height(tblMFgt0)==height(tblMFgt),...
          'Specified ''tblMFgt'' contains unlabeled row(s).');
        assert(isequal(tblMFgt(:,MFTable.FLDSID),tblMFgt0));
        assert(isa(tblMFgt.mov,'MovieIndex'));
      else
        % tblMFgt supplied, and should have labels etc.
      end
      assert(isa(tblMFgt.mov,'MovieIndex'));
      [~,gt] = tblMFgt.mov.get();
      assert(~any(gt));
      
      if ~tfPart
        movC = categorical(tblMFgt.mov);
        tgtC = categorical(tblMFgt.iTgt);
        grpC = movC.*tgtC;
        cvPart = cvpartition(grpC,'kfold',kFold);
        partTrn = arrayfun(@(x)cvPart.training(x),1:kFold,'uni',0);
        partTst = arrayfun(@(x)cvPart.test(x),1:kFold,'uni',0);
        partTrn = cat(2,partTrn{:});
        partTst = cat(2,partTst{:});
      else
        partTrn = ~partTst;
      end
      assert(islogical(partTrn) && islogical(partTst));
      n = height(tblMFgt);
      szassert(partTrn,[n kFold]);
      szassert(partTst,[n kFold]);
      tmp = partTrn+partTst;
      assert(all(tmp(:)==1),'Invalid cv splits specified.'); % partTrn==~partTst
      assert(all(sum(partTst,2)==1),...
        'Invalid cv splits specified; each row must be tested precisely once.');
      
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:tracker','No tracker is available for this project.');
      end
      if ~strcmp(tObj.algorithmName,'cpr')
        % DeepTrackers do non-blocking/bg tracking
        error('Only CPR tracking currently supported.');
      end      

      if ~dontInitH0
        obj.preProcUpdateH0IfNec();
      end
      
      % Basically an initHook() here
      if initData
        obj.preProcInitData();
        obj.ppdbInit();
      end
      tObj.trnDataInit(); % not strictly necessary as .retrain() should do it 
      tObj.trnResInit(); % not strictly necessary as .retrain() should do it 
      tObj.trackResInit();
      tObj.vizInit();
      tObj.asyncReset();
      
      npts = obj.nLabelPoints;
      pTrkCell = cell(kFold,1);
      dGTTrkCell = cell(kFold,1);
      if tfWB
        wbObj.startPeriod('Fold','shownumden',true,'denominator',kFold);
      end
      for iFold=1:kFold
        if tfWB
          wbObj.updateFracWithNumDen(iFold);
        end
        tblMFgtTrain = tblMFgt(partTrn(:,iFold),:);
        tblMFgtTrack = tblMFgt(partTst(:,iFold),:);
        fprintf(1,'Fold %d: nTrain=%d, nTest=%d.\n',iFold,...
          height(tblMFgtTrain),height(tblMFgtTrack));
        if tfWB
          wbObj.startPeriod('Training','nobar',true);
        end
        tObj.retrain('tblPTrn',tblMFgtTrain,'wbObj',wbObj);
        if tfWB
          wbObj.endPeriod();
        end
        tObj.track(tblMFgtTrack,'wbObj',wbObj);        
        [tblTrkRes,pTrkiPt] = tObj.getAllTrackResTable(); % if wbObj.isCancel, partial tracking results
        if initData
          obj.preProcInitData();
          obj.ppdbInit();
        end
        tObj.trnDataInit();
        tObj.trnResInit();
        tObj.trackResInit();
        if tfWB && wbObj.isCancel
          return;
        end
        
        assert(isequal(pTrkiPt(:)',1:npts));
        assert(isequal(tblTrkRes(:,MFTable.FLDSID),...
                       tblMFgtTrack(:,MFTable.FLDSID)));
        if obj.hasTrx || obj.cropProjHasCrops
          pGT = tblMFgtTrack.pAbs;
        else
          if tblfldscontains(tblMFgtTrack,'pAbs')
            assert(isequal(tblMFgtTrack.p,tblMFgtTrack.pAbs));
          end
          pGT = tblMFgtTrack.p;
        end
        d = tblTrkRes.pTrk - pGT;
        [ntst,Dtrk] = size(d);
        assert(Dtrk==npts*2); % npts=nPhysPts*nview
        d = reshape(d,ntst,npts,2);
        d = sqrt(sum(d.^2,3)); % [ntst x npts]
        
        pTrkCell{iFold} = tblTrkRes;
        dGTTrkCell{iFold} = d;
      end

      % create output table
      for iFold=1:kFold
        tblFold = table(repmat(iFold,height(pTrkCell{iFold}),1),...
          'VariableNames',{'fold'});
        pTrkCell{iFold} = [tblFold pTrkCell{iFold}];
      end
      pTrkAll = cat(1,pTrkCell{:});
      dGTTrkAll = cat(1,dGTTrkCell{:});
      assert(isequal(height(pTrkAll),height(tblMFgt),size(dGTTrkAll,1)));
      [tf,loc] = tblismember(tblMFgt,pTrkAll,MFTable.FLDSID);
      assert(all(tf));
      pTrkAll = pTrkAll(loc,:);
      dGTTrkAll = dGTTrkAll(loc,:);
      
      if tblfldscontains(tblMFgt,'roi')
        flds = MFTable.FLDSCOREROI;
      else
        flds = MFTable.FLDSCORE;
      end
      tblXVres = tblMFgt(:,flds);
      if tblfldscontains(tblMFgt,'pAbs')
        tblXVres.p = tblMFgt.pAbs;
      end
      tblXVres.pTrk = pTrkAll.pTrk;
      tblXVres.dGTTrk = dGTTrkAll;
      tblXVres = [pTrkAll(:,'fold') tblXVres];
      
      obj.xvResults = tblXVres;
      obj.xvResultsTS = now;
    end
    
    function tblRes = trackTrainTrackEval(obj,tblMFTtrn,tblMFTtrk,varargin)
      % Like single train/track crossvalidation "fragment"
      %
      % tblMFTtrn: MFTable, MFTable.FLDSID only
      % tblMFTtrk: etc
      %
      % tblRes: table of tracking err/results      
      
      wbObj = myparse(varargin,...
        'wbObj',[]... % (opt) WaitBarWithCancel
      );        
      
      tfWB = ~isempty(wbObj);
      
      tblfldsassert(tblMFTtrn,MFTable.FLDSID);
      tblfldsassert(tblMFTtrk,MFTable.FLDSID);
                  
      tObj = obj.tracker;
      if isempty(tObj)
        error('Labeler:tracker','No tracker is available for this project.');
      end
      if ~strcmp(tObj.algorithmName,'cpr')
        % DeepTrackers do non-blocking/bg tracking
        error('Only CPR tracking currently supported.');
      end      

      obj.preProcUpdateH0IfNec();

      % codepath requires CPR; allow 'treatInfPosAsOcc' to default to false 
      tblMFTPtrn = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFTtrn);
      tblMFTtrk = obj.preProcGetMFTableLbled('tblMFTrestrict',tblMFTtrk);
      
      tObj.trackResInit();
      tObj.vizInit();
      tObj.asyncReset();
      tObj.retrain('tblPTrn',tblMFTPtrn,'wbObj',wbObj);
      if tfWB && wbObj.isCancel
        % tracker obj responsible for these things
        % tObj.trnDataInit();
        % tObj.trnResInit();
        tblRes = [];        
        return;
      end
      
      tObj.track(tblMFTtrk(:,MFTable.FLDSID),'wbObj',wbObj);      
      if tfWB && wbObj.isCancel
        tObj.trnDataInit();
        tObj.trnResInit();
        tObj.trackResInit();
        tObj.vizInit();
        tblRes = [];        
        return;
      end      
      
      [tblTrkRes,pTrkiPt] = tObj.getAllTrackResTable();
      tObj.trnDataInit();
      tObj.trnResInit();
      tObj.trackResInit();
      tObj.vizInit();
        
      npts = obj.nLabelPoints;      
      assert(isequal(pTrkiPt(:)',1:npts));
      assert(isequal(tblTrkRes(:,MFTable.FLDSID),...
        tblMFTtrk(:,MFTable.FLDSID)));
      if obj.hasTrx || obj.cropProjHasCrops
        pGT = tblMFTtrk.pAbs;
      else
        tblfldsdonotcontainassert(tblMFTtrk,'pAbs');
        pGT = tblMFTtrk.p;
      end
      d = tblTrkRes.pTrk - pGT;
      [ntst,Dtrk] = size(d);
      assert(Dtrk==npts*2); % npts=nPhysPts*nview
      d = reshape(d,ntst,npts,2);
      d = sqrt(sum(d.^2,3)); % [ntst x npts]
      
      tblRes = tblTrkRes;
      tblRes.pLbl = pGT;
      tblRes.dLblTrk = d;
      
%       if tblfldscontains(tblMFgt,'roi')
%         flds = MFTable.FLDSCOREROI;
%       else
%         flds = MFTable.FLDSCORE;
%       end
%       tblXVres = tblMFgt(:,flds);
%       if tblfldscontains(tblMFgt,'pAbs')
%         tblXVres.p = tblMFgt.pAbs;
%       end
%       tblXVres.pTrk = pTrkAll.pTrk;
%       tblXVres.dGTTrk = dGTTrkAll;
%       tblXVres = [pTrkAll(:,'fold') tblXVres];      
    end
    
    function trackCrossValidateVizPrctiles(obj,tblXVres,varargin)
      ptiles = myparse(varargin,...
        'prctiles',[50 75 90 95]);
      
      assert(~obj.gtIsGTMode,'Not supported in GT mode.');
      
      err = tblXVres.dGTTrk;
      npts = obj.nLabelPoints;
      nphyspts = obj.nPhysPoints;
      assert(size(err,2)==npts);
      nptiles = numel(ptiles);
      ptl = prctile(err,ptiles,1)';
      szassert(ptl,[npts nptiles]);
      
      % for now always viz with first row
      iVizRow = 1;
      vizrow = tblXVres(iVizRow,:);
      xyLbl = Shape.vec2xy(vizrow.p);
      szassert(xyLbl,[npts 2]);
      tfRoi = tblfldscontains(vizrow,'roi');
      if tfRoi 
        [xyLbl,tfOOBview] = Shape.xy2xyROI(xyLbl,vizrow.roi,nphyspts);
        if any(tfOOBview)
          warningNoTrace('One or more landmarks lies outside ROI.');
        end
      end
      for ivw=1:obj.nview
        mr = MovieReader;
        obj.movieMovieReaderOpen(mr,MovieIndex(vizrow.mov),ivw);
        im = mr.readframe(vizrow.frm,'docrop',false); 
        % want to crop via vizrow.roi (if applicable); this comes from trx
        % most of the time
        iptsvw = (1:nphyspts)' + nphyspts*(ivw-1);
        xyLblVw = xyLbl(iptsvw,:); 
        ptlVw = ptl(iptsvw,:);
        if tfRoi
          roiVw = vizrow.roi((1:4)+(ivw-1)*4);
          im = padgrab2(im,0,roiVw(3),roiVw(4),roiVw(1),roiVw(2));
        end
      
        if obj.nview==1
          figure('Name','Tracker performance');
        else
          figure('Name',sprintf('Tracker performance (view %d)',ivw));
        end          
        imagesc(im);
        colormap gray;
        hold on;
        clrs = hsv(nptiles)*.75;
        for ipt=1:nphyspts
          for iptile=1:nptiles        
            r = ptlVw(ipt,iptile);
            pos = [xyLblVw(ipt,:)-r 2*r 2*r];
             rectangle('position',pos,'curvature',1,'edgecolor',clrs(iptile,:));
          end
        end
        axis image ij
        grid on;
        if ivw==1
          tstr = sprintf('Cross-validation ptiles: %s. n=%d, mean err=%.3fpx.',...
            mat2str(ptiles),height(tblXVres),mean(err(:)));
          title(tstr,'fontweight','bold','interpreter','none');
          hLeg = arrayfun(@(x)plot(nan,nan,'-','Color',clrs(x,:)),1:nptiles,...
            'uni',0);      
          legend([hLeg{:}],arrayfun(@num2str,ptiles,'uni',0));
        end
      end
    end
    
    function [tf,lposTrk,occTrk] = trackIsCurrMovFrmTracked(obj,iTgt)
      % tf: scalar logical, true if tracker has results/predictions for 
      %   currentMov/frm/iTgt 
      % lposTrk: [nptsx2] if tf is true, xy coords from tracker; otherwise
      %   indeterminate
      % occTrk: [npts] if tf is true, point is predicted as occluded
      
      tObj = obj.tracker;
      if isempty(tObj)
        tf = false;
        lposTrk = [];
        occTrk = [];
      else
        if isa(tObj,'CPRLabelTracker')
          xy = tObj.getPredictionCurrentFrame(); % [nPtsx2xnTgt]
          occ = false(obj.nLabelPoints,obj.nTargets);
        else
          [xy,occ] = tObj.getPredictionCurrentFrame();
        end
        szassert(xy,[obj.nLabelPoints 2 obj.nTargets]);
        
        lposTrk = xy(:,:,iTgt);
        occTrk = occ(:,iTgt);
        tfnan = isnan(xy); % hmm wonder why look at all of xy rather than just lposTrk
        tf = any(~tfnan(:));
      end
    end
    
    function trackLabelMontage(obj,tbl,errfld,varargin)
      [nr,nc,h,npts,nphyspts,nplot,frmlblclr,frmlblbgclr] = myparse(varargin,...
        'nr',3,...
        'nc',4,...
        'hPlot',[],...
        'npts',obj.nLabelPoints,... % hack
        'nphyspts',obj.nPhysPoints,... % hack
        'nplot',height(tbl),... % show/include nplot worst rows
        'frmlblclr',[1 1 1], ...
        'frmlblbgclr',[0 0 0] ...
        );
      
      if nplot>height(tbl)
        warningNoTrace('''nplot'' argument too large. Only %d GT rows are available.',height(tbl));
        nplot = height(tbl);
      end
      
      tbl = sortrows(tbl,{errfld},{'descend'});
      tbl = tbl(1:nplot,:);
      
      tbl = obj.preProcCropLabelsToRoiIfNec(tbl,...
        'doRemoveOOB',false,...
        'pfld','pLbl',...
        'pabsfld','pLblAbs',...
        'proifld','pLblRoi');
      tbl = obj.preProcCropLabelsToRoiIfNec(tbl,...
        'doRemoveOOB',false,...
        'pfld','pTrk',...
        'pabsfld','pTrkAbs',...
        'proifld','pTrkRoi');

      % tbl.pLbl/pTrk now in relative coords if appropriate
     
      % Create a table to call preProcDataFetch so we can use images in
      % preProc cache.      
      FLDSTMP = {'mov' 'frm' 'iTgt' 'tfoccLbl' 'pLbl'}; % MFTable.FLDSCORE
      tfROI = tblfldscontains(tbl,'roi');     
      if tfROI
        FLDSTMP = [FLDSTMP 'roi'];
      end
%       if tblisfield(tbl,'nNborMask')
%         % Why this is in the cache?
%         FLDSTMP = [FLDSTMP 'nNborMask'];
%       end
      tblCacheUpdate = tbl(:,FLDSTMP);
      tblCacheUpdate.Properties.VariableNames(4:5) = {'tfocc' 'p'};
      %tblCacheUpdate.p = pLbl; % corrected for ROI if nec
      [ppdata,ppdataIdx,~,~,tfReadFailed] = ...
        obj.preProcDataFetch(tblCacheUpdate,'updateRowsMustMatch',true);
      nReadFailed = nnz(tfReadFailed);
      if nReadFailed>0
        warningNoTrace('Failed to read %d frames/images; these will not be included in montage.',...
          nReadFailed);
        % Would be better to include with "blank" image
      end
      
      I = ppdata.I(ppdataIdx,:);    
      tblPostRead = tbl(:,{'pLbl' 'pTrk' 'mov' 'frm' 'iTgt' errfld});
      tblPostRead(tfReadFailed,:) = [];
    
      if obj.hasTrx
        frmLblsAll = arrayfun(@(zm,zf,zt,ze)sprintf('mov/frm/tgt=%d/%d/%d,err=%.2f',zm,zf,zt,ze),...
          abs(tblPostRead.mov),tblPostRead.frm,tblPostRead.iTgt,tblPostRead.(errfld),'uni',0);        
      else
        frmLblsAll = arrayfun(@(zm,zf,ze)sprintf('mov/frm=%d/%d,err=%.2f',zm,zf,ze),...
          abs(tblPostRead.mov),tblPostRead.frm,tblPostRead.(errfld),'uni',0);
      end
      
      nrowsPlot = height(tblPostRead);
      startIdxs = 1:nr*nc:nrowsPlot;
      for i=1:numel(startIdxs)
        plotIdxs = startIdxs(i):min(startIdxs(i)+nr*nc-1,nrowsPlot);
        frmLblsThis = frmLblsAll(plotIdxs);
        for iView=1:obj.nview
          h(end+1,1) = figure('Name','Tracking Error Montage','windowstyle','docked'); %#ok<AGROW>
          pColIdx = (1:nphyspts)+(iView-1)*nphyspts;
          pColIdx = [pColIdx pColIdx+npts]; %#ok<AGROW>
          Shape.montage(I(:,iView),tblPostRead.pLbl(:,pColIdx),'fig',h(end),...
            'nr',nr,'nc',nc,'idxs',plotIdxs,...
            'framelbls',frmLblsThis,'framelblscolor',frmlblclr,...
            'framelblsbgcolor',frmlblbgclr,'p2',tblPostRead.pTrk(:,pColIdx),...
            'p2marker','+','titlestr','Tracking Montage, descending err (''+'' is tracked)');
        end
      end
    end
  end
  methods (Static)
    function tblLbled = hlpTblLbled(tblLbled)
      tblLbled.mov = int32(tblLbled.mov);
      tblLbled = tblLbled(:,[MFTable.FLDSID {'p'}]);
      isLbled = true(height(tblLbled),1);
      tblLbled = [tblLbled table(isLbled)];
    end
    function trkErr = hlpTblErr(tblBig,fldLbled,fldTrked,fldpLbl,fldpTrk,npts)
      tf = tblBig.(fldLbled) & tblBig.(fldTrked);
      pLbl = tblBig.(fldpLbl)(tf,:);
      pTrk = tblBig.(fldpTrk)(tf,:);
      szassert(pLbl,size(pTrk));
      nTrkLbl = size(pLbl,1);
      dErr = pLbl-pTrk;
      dErr = reshape(dErr,nTrkLbl,npts,2);
      dErr = sqrt(sum(dErr.^2,3)); % [nTrkLbl x npts] L2 err
      dErr = mean(dErr,2); % [nTrkLblx1] L2 err, mean across pts      
      trkErr = nan(height(tblBig),1);
      trkErr(tf) = dErr;
    end
    
    function sPrm = trackGetParamsFromStruct(s)
      % Get all parameters:
      %  - preproc
      %  - cpr
      %  - common dl
      %  - specific dl
      %
      % sPrm: scalar struct containing NEW-style params:
      % sPrm.ROOT.Track
      %          .CPR
      %          .DeepTrack (if applicable)
      % Top-level fields .Track, .CPR, .DeepTrack may be missing if they
      % don't exist yet.
      
      % Future TODO: As in trackSetParams, currently this is hardcoded when
      % it ideally would just be a generic loop
      
      if isfield(s,'trackParams'),
        sPrm = s.trackParams;
        return;
      end
      
      prmCpr = [];
      for iTrk=1:numel(s.trackerData)
        if     strcmp(s.trackerClass{iTrk}{1},'CPRLabelTracker') ...
            && ~isempty(s.trackerData{iTrk})
          prmCpr = s.trackerData{iTrk}.sPrm;
          break;
        end
      end
      
      prmPP = s.preProcParams;
      
      prmDLCommon = s.trackDLParams;
      
      prmDLSpecific = struct;
      for i = 1:numel(s.trackerData),
        if ~strcmp(s.trackerClass{i}{1},'DeepTracker') || isempty(s.trackerData{i}),
          continue;
        end
        trnNetType = s.trackerData{i}.trnNetType.prettyString;
        prmDLSpecific.(trnNetType) = s.trackerData{i}.sPrm;
      end
      
      prmTrack = struct;
      prmTrack.trackNFramesSmall = s.cfg.Track.PredictFrameStep;
      prmTrack.trackNFramesLarge = s.cfg.Track.PredictFrameStepBig;
      prmTrack.trackNFramesNear = s.cfg.Track.PredictNeighborhood;
      
      sPrm = Labeler.trackGetParamsHelper(prmCpr,prmPP,prmDLCommon,...
                                          prmDLSpecific,prmTrack);      
    end
    
    function sPrmAll = trackGetParamsHelper(prmCpr,prmPP,prmDLCommon,prmDLSpecific,obj)
      
      sPrmAll = APTParameters.defaultParamsStructAll;
      
%      assert(~xor(isempty(prmCpr),isempty(prmPP)));
      if ~isempty(prmCpr)
        sPrmAll = APTParameters.setCPRParams(sPrmAll,prmCpr);
      end
      if ~isempty(prmPP),
        sPrmAll = APTParameters.setPreProcParams(sPrmAll,prmPP);
      end
      if ~isempty(obj)
        sPrmAll = APTParameters.setNFramesTrackParams(sPrmAll,obj);
      end
      if ~isempty(prmDLCommon)
        sPrmAll = APTParameters.setTrackDLParams(sPrmAll,prmDLCommon);
      end
            
      % specific parameters
      fns = fieldnames(prmDLSpecific);
      for i = 1:numel(fns),
        sPrmAll = APTParameters.setDLSpecificParams(sPrmAll,fns{i},prmDLSpecific.(fns{i}));
      end
      
    end
    
  end
  methods    
    function tblBig = trackGetBigLabeledTrackedTable(obj,varargin)
      % tbl: MFT table indcating isLbled, isTrked, trkErr, etc.
      
      wbObj = myparse(varargin,...
        'wbObj',[]); % optional WaitBarWithContext. If .isCancel:
                     % 1. tblBig is indeterminate
                     % 2. obj should be logically const
      tfWB = ~isempty(wbObj);
      
      tblLbled = obj.labelGetMFTableLabeled('wbObj',wbObj);
      if tfWB && wbObj.isCancel
        tblBig = [];
        return;
      end      
      tblLbled = Labeler.hlpTblLbled(tblLbled);
      
      tblLbled2 = obj.labelGetMFTableLabeled('wbObj',wbObj,'useLabels2',true);
      if tfWB && wbObj.isCancel
        tblBig = [];
        return;
      end

      tblLbled2 = Labeler.hlpTblLbled(tblLbled2);
      tblfldsassert(tblLbled2,[MFTable.FLDSID {'p' 'isLbled'}]);
      tblLbled2.Properties.VariableNames(end-1:end) = {'pImport' 'isImported'};
      
      npts = obj.nLabelPoints;

      tObj = obj.tracker;
      if ~isempty(tObj)
        tblTrked = tObj.getAllTrackResTable();
      else
        tblTrked = [];
      end
      if ~isempty(tblTrked)
%         tblTrked = tObj.trkPMD(:,MFTable.FLDSID);
        tblTrked.mov = int32(tblTrked.mov); % from MovieIndex
%         pTrk = tblTrked.pTrk;
%         if isempty(pTrk)
%           % AL 20180620 Shouldn't be nec. 
%           % edge case, tracker not initting properly
%           pTrk = nan(0,npts*2);
%         end
        isTrked = true(height(tblTrked),1);
        tblTrked = [tblTrked table(isTrked)];     
      end

      if isempty(tblTrked)
        % 20171106 ML bug 2015b outerjoin empty input table
        % Seems fixed in 16b
        pTrk = nan(size(tblLbled.p));
        isTrked = false(size(tblLbled.isLbled));
        tblBig = [tblLbled table(pTrk,isTrked)];
      else
        tblBig = outerjoin(tblLbled,tblTrked,'Keys',MFTable.FLDSID,...
          'MergeKeys',true);
      end
      tblBig = outerjoin(tblBig,tblLbled2,'Keys',MFTable.FLDSID,'MergeKeys',true);
        
      % Compute tracking err (for rows both labeled and tracked)
      npts = obj.nLabelPoints;
      trkErr = Labeler.hlpTblErr(tblBig,'isLbled','isTrked','p','pTrk',npts);
      importedErr = Labeler.hlpTblErr(tblBig,'isLbled','isImported','p','pImport',npts); % treating as imported tracking
      tblBig = [tblBig table(trkErr) table(importedErr)];
      
      xvres = obj.xvResults;
      tfhasXV = ~isempty(xvres);
      if tfhasXV
        tblXVerr = rowfun(@nanmean,xvres,'InputVariables',{'dGTTrk'},...
          'OutputVariableNames',{'xvErr'});
        hasXV = true(height(xvres),1);
        tblXV = [xvres(:,MFTable.FLDSID) tblXVerr table(hasXV)];        
      else
        tblXV = table(nan(0,1),nan(0,1),nan(0,1),nan(0,1),false(0,1),...
          'VariableNames',{'mov' 'frm' 'iTgt' 'xvErr' 'hasXV'});
      end
      
      % 20171106 ML bug 2015b outerjoin empty input table
      tblBig = outerjoin(tblBig,tblXV,'Keys',MFTable.FLDSID,'MergeKeys',true);
      tblBig = tblBig(:,[MFTable.FLDSID {'trkErr' 'importedErr' 'xvErr' 'isLbled' 'isTrked' 'isImported' 'hasXV'}]);
      if tfhasXV && ~isequal(tblBig.isLbled,tblXV.hasXV)
        warningNoTrace('Cross-validation results appear out-of-date with respect to current set of labeled frames.');
      end
    end
    
    function tblSumm = trackGetSummaryTable(obj,tblBig)
      % tblSumm: Big summary table, one row per (mov,tgt)
      
      assert(~obj.gtIsGTMode,'Currently unsupported in GT mode.');
      
      % generate tblSummBase
      if obj.hasTrx
        assert(obj.nview==1,'Currently unsupported for multiview projects.');
        tfaf = obj.trxFilesAllFull;
        dataacc = nan(0,4); % mov, tgt, trajlen, frm1
        for iMov=1:obj.nmovies
          tfile = tfaf{iMov,1};
          tifo = obj.trxCache(tfile);
          frm2trxI = tifo.frm2trx;

          nTgt = size(frm2trxI,2);
          for iTgt=1:nTgt
            tflive = frm2trxI(:,iTgt);
            sp = get_interval_ends(tflive);
            if isempty(sp)
              trajlen = 0;
              frm1 = nan;
            else
              if numel(sp)>1
                warningNoTrace('Movie %d, target %d is live over non-consecutive frames.',...
                  iMov,iTgt);
              end
              trajlen = nnz(tflive); % when numel(sp)>1, track is 
              % non-consecutive and this won't strictly be trajlen
              frm1 = sp(1);
            end
            dataacc(end+1,:) = [iMov iTgt trajlen frm1]; %#ok<AGROW>
          end
        end
      
        % Contains every (mov,tgt)
        tblSummBase = array2table(dataacc,'VariableNames',{'mov' 'iTgt' 'trajlen' 'frm1'});
      else
        mov = (1:obj.nmovies)';
        iTgt = ones(size(mov));
        nfrms = cellfun(@(x)x.nframes,obj.movieInfoAll(:,1));
        trajlen = nfrms;
        frm1 = iTgt;
        tblSummBase = table(mov,iTgt,trajlen,frm1);
      end
      
      tblStats = rowfun(@Labeler.hlpTrackTgtStats,tblBig,...
        'InputVariables',{'isLbled' 'isTrked' 'isImported' 'hasXV' 'trkErr' 'importedErr' 'xvErr'},...
        'GroupingVariables',{'mov' 'iTgt'},...
        'OutputVariableNames',{'nFrmLbl' 'nFrmTrk' 'nFrmImported' ...
          'nFrmLblTrk' 'lblTrkMeanErr' ...
          'nFrmLblImported' 'lblImportedMeanErr' ...
          'nFrmXV' 'xvMeanErr'},...
        'NumOutputs',9);

      [tblSumm,iTblSummB,iTblStats] = outerjoin(tblSummBase,tblStats,...
        'Keys',{'mov' 'iTgt'},'MergeKeys',true);
      assert(~any(iTblSummB==0)); % tblSummBase should contain every (mov,tgt)
      tblSumm(:,{'GroupCount'}) = [];
      tfNoStats = iTblStats==0;
      % These next 4 sets replace NaNs from the join (due to "no
      % corresponding row in tblStats") with zeros
      tblSumm.nFrmLbl(tfNoStats) = 0;
      tblSumm.nFrmTrk(tfNoStats) = 0;
      tblSumm.nFrmImported(tfNoStats) = 0;
      tblSumm.nFrmLblTrk(tfNoStats) = 0;
      tblSumm.nFrmLblImported(tfNoStats) = 0;
      tblSumm.nFrmXV(tfNoStats) = 0;
    end  
  end
  methods (Static)
      function [nFrmLbl,nFrmTrk,nFrmImported,...
          nFrmLblTrk,lblTrkMeanErr,...
          nFrmLblImported,lblImportedMeanErr,...
          nFrmXV,xvMeanErr] = ...
          hlpTrackTgtStats(isLbled,isTrked,isImported,hasXV,...
          trkErr,importedErr,xvErr)
        % nFrmLbl: number of labeled frames
        % nFrmTrk: number of tracked frames
        % nFrmLblTrk: " labeled&tracked frames
        % lblTrkMeanErr: L2 err, mean across pts, mean over labeled&tracked frames
        % nFrmXV: number of frames with XV res (should be same as nFrmLbl unless XV
        %   out-of-date)
        % xvMeanErr: L2 err, mean across pts, mean over xv frames
        
        nFrmLbl = nnz(isLbled);
        nFrmTrk = nnz(isTrked);
        nFrmImported = nnz(isImported);
        
        tfLbledTrked = isLbled & isTrked;
        nFrmLblTrk = nnz(tfLbledTrked);
        lblTrkMeanErr = nanmean(trkErr(tfLbledTrked));
        
        tfLbledImported = isLbled & isImported;
        nFrmLblImported = nnz(tfLbledImported);
        lblImportedMeanErr = nanmean(importedErr(tfLbledImported));        
        
        nFrmXV = nnz(hasXV);
        xvMeanErr = mean(xvErr(hasXV));
      end
      
 
%     function [success,fname] = trackSaveLoadAsHelper(obj,rcprop,uifcn,...
%         promptstr,rawMeth)
%       % rcprop: Name of RC property for guessing path
%       % uifcn: either 'uiputfile' or 'uigetfile'
%       % promptstr: used in uiputfile
%       % rawMeth: track*Raw method to call when a file is specified
%       
%       % Guess a path/location for save/load
%       lastFile = RC.getprop(rcprop);
%       if isempty(lastFile)
%         projFile = obj.projectfile;
%         if ~isempty(projFile)
%           savepath = fileparts(projFile);
%         else
%           savepath = pwd;
%         end
%       else
%         savepath = fileparts(lastFile);
%       end
%       
%       filterspec = fullfile(savepath,'*.mat');
%       [fname,pth] = feval(uifcn,filterspec,promptstr);
%       if isequal(fname,0)
%         fname = [];
%         success = false;
%       else
%         fname = fullfile(pth,fname);
%         success = true;
%         obj.(rawMeth)(fname);
%       end
%     end
     
  end
  
  methods % TrackRes
    
    function trackResInit(obj)
      obj.trkResIDs = cell(0,1);
      obj.trkRes = cell(obj.nmovies,obj.nview,0);
      obj.trkResGT = cell(obj.nmoviesGT,obj.nview,0);
      for i=1:numel(obj.trkResViz)
        delete(obj.trkResViz{i});
      end
      obj.trkResViz = cell(0,1);
    end
    
    function [tf,iTrkRes] = trackResFindID(obj,id)
      iTrkRes = find(strcmp(obj.trkResIDs,id));
      assert(isempty(iTrkRes) || isscalar(iTrkRes));
      tf = ~isempty(iTrkRes);      
    end
    
    function tv = trackResFindTV(obj,id) % throws
      % Return TrackingVisualizer object for id, or err
      [tf,iTR] = obj.trackResFindID(id);
      if ~tf
        error('Tracking results ''%s'' not found in project.',id);
      end
      tv = obj.trkResViz{iTR};
    end
    
    function iTrkRes = trackResEnsureID(obj,id)
      iTrkRes = find(strcmp(obj.trkResIDs,id));
      if isempty(iTrkRes)
        handleTagPfix = ['handletag_' id];
        tv = TrackingVisualizer(obj,handleTagPfix);
        tv.vizInit();
        
        fprintf(1,'Adding new tracking results set ''%s''.\n',id);
        obj.trkResIDs{end+1,1} = id;
        obj.trkRes(:,:,end+1) = {[]};
        obj.trkResGT(:,:,end+1) = {[]};
        obj.trkResViz{end+1,1} = tv;
        iTrkRes = numel(obj.trkResIDs);
      end
    end
    
    function trackResRmID(obj,id)
      iTR = find(strcmp(obj.trkResIDs,id));
      if isempty(iTR)
        error('Tracking results ID ''%s'' not found in project',id);
      end

      assert(isscalar(iTR));
      obj.trkResIDs(iTR,:) = [];
      obj.trkRes(:,:,iTR) = [];
      obj.trkResGT(:,:,iTR) = [];
      tv = obj.trkResViz(iTR);
      delete(tv);
      obj.trkResViz(iTR,:) = [];
    end
    
    function trackResAddCurrMov(obj,id,trkfiles)
      if ~obj.hasMovie
        error('No movie is open.');
      end
      mIdx = obj.currMovIdx;
      obj.trackResAdd(id,mIdx,trkfiles);
    end
    
    function trackResAdd(obj,id,mIdx,trkfiles)
      % id: trackRes ID
      % mIdx: [n] MovieIndex vector
      % trkfiles: [nxnview] cell of trkfile objects or fullpaths

      assert(isa(mIdx,'MovieIndex'));
      n = numel(mIdx);
      if ischar(trkfiles)
        trkfiles = cellstr(trkfiles);
      end
      assert(iscell(trkfiles));
      szassert(trkfiles,[n obj.nview]);
      
      iTR = obj.trackResEnsureID(id);
      [iMovs,gt] = mIdx.get();
      if gt
        TRFLD = 'trkResGT';
      else
        TRFLD = 'trkRes';
      end
      
      if iscellstr(trkfiles) || isstring(trkfiles)
        trkfileobjs = cellfun(@TrkFile.load,trkfiles,'uni',0);
      else
        assert(isa(trkfiles{1},'TrkFile'));
        trkfileobjs = trkfiles;
      end
      
      for iMov = iMovs(:)'
        v = obj.(TRFLD){iMov,1,iTR};
        if ~isempty(v)
          warningNoTrace('Overwriting existing tracking results for ''%s'' at movie index %d.',...
            id,iMov);
        end
      end
      obj.(TRFLD)(iMovs,:,iTR) = trkfileobjs;      
    end
    
    function hlpTrackResSetViz(obj,tvMeth,id,varargin)
      if isempty(id)
        cellfun(@(x)x.(tvMeth)(varargin{:}),obj.trkResViz);
      else
        tv = obj.trackResFindTV(id); % throws
        tv.(tvMeth)(varargin{:});
      end
    end
    
    function trackResSetHideViz(obj,id,tf)
      obj.hlpTrackResSetViz('setHideViz',id,tf);
    end
    
    function trackResSetHideTextLbls(obj,id,tf)
      obj.hlpTrackResSetViz('setHideTextLbls',id,tf);
    end    
    
    function trackResSetMarkerCosmetics(obj,id,varargin)
      obj.hlpTrackResSetViz('setMarkerCosmetics',id,varargin);
    end

    function trackResSetTextCosmetics(obj,id,varargin)
      obj.hlpTrackResSetViz('setTextCosmetics',id,varargin);
    end

  end
   
  %% Video
  methods
    
    function videoCenterOnCurrTarget(obj)
      % Shift axis center/target and CameraUpVector without touching zoom.
      % 
      % Potential TODO: CamViewAngle treatment looks a little bizzare but
      % seems to work ok. Theoretically better (?), at movieSet time, cache
      % a default CameraViewAngle, and at movieRotateTargetUp set time, set
      % the CamViewAngle to either the default or the default/2 etc.

      [x0,y0] = obj.videoCurrentCenter();
      [x,y,th] = obj.currentTargetLoc();

      dx = x-x0;
      dy = y-y0;
      ax = obj.gdata.axes_curr;
      axisshift(ax,dx,dy);
      ax.CameraPositionMode = 'auto'; % issue #86, behavior differs between 16b and 15b. Use of manual zoom toggles .CPM into manual mode
      ax.CameraTargetMode = 'auto'; % issue #86, etc Use of manual zoom toggles .CTM into manual mode
      %ax.CameraViewAngleMode = 'auto';
      if obj.movieRotateTargetUp
        ax.CameraUpVector = [cos(th) sin(th) 0];
        if verLessThan('matlab','R2016a')
          % See iss#86. In R2016a, the zoom/pan behavior of axes in 3D mode
          % (currently, any axis with CameraViewAngleMode manually set)
          % changed. Prior to R2016a, zoom on such an axis altered camera
          % position via .CameraViewAngle, etc, with the axis limits
          % unchanged. Starting in R2016a, zoom on 3D axes changes the axis
          % limits while the camera position is unchanged.
          %
          % Currently we prefer the modern treatment and the
          % center-on-target, rotate-target, zoom slider, etc treatments
          % are built around that treatment. For prior MATLABs, we work
          % around -- it is a little awkward as the fundamental strategy
          % behind zoom is different. For prior MATLABs users should prefer
          % the Zoom slider in the Targets panel as opposed to using the
          % zoom tools in the toolbar.
          hF = obj.gdata.figure;
          tf = getappdata(hF,'manualZoomOccured');
          if tf
            ax.CameraViewAngleMode = 'auto';
            setappdata(hF,'manualZoomOccured',false);
          end
        end
        if strcmp(ax.CameraViewAngleMode,'auto')
          cva = ax.CameraViewAngle;
          ax.CameraViewAngle = cva/2;
        end
      else
        ax.CameraUpVectorMode = 'auto';
      end
    end
    
    function videoCenterOnCurrTargetPoint(obj)
      [tfsucc,xy] = obj.videoCenterOnCurrTargetPointHelp();
      if tfsucc
        [x0,y0] = obj.videoCurrentCenter;
        dx = xy(1)-x0;
        dy = xy(2)-y0;
        ax = obj.gdata.axes_curr;
        axisshift(ax,dx,dy);
        ax.CameraPositionMode = 'auto'; % issue #86, behavior differs between 16b and 15b. Use of manual zoom toggles .CPM into manual mode
        ax.CameraTargetMode = 'auto'; % issue #86, etc Use of manual zoom toggles .CTM into manual mode
        %ax.CameraViewAngleMode = 'auto';
      end
    end
    
    function [tfsucc,xy] = videoCenterOnCurrTargetPointHelp(obj)
      % get (x,y) for current movieCenterOnTargetIPt
      
      tfsucc = true;
      f = obj.currFrame;
      itgt = obj.currTarget;
      ipt = obj.movieCenterOnTargetIpt;
      
      lpos = obj.labeledposCurrMovie;
      if ~isempty(lpos)
        xy = lpos(ipt,:,f,itgt);
        if all(~isnan(xy))
          return;
        end
      end
      
      tracker = obj.tracker;
      if ~isempty(tracker)
        tpos = tracker.getTrackingResultsCurrMovie;
        if ~isempty(tpos)
          xy = tpos(ipt,:,f,itgt);
          if all(~isnan(xy))
            return;
          end
        end
      end
      
      lpos2 = obj.labeledpos2CurrMovie;
      if ~isempty(lpos2)
        xy = lpos2(ipt,:,f,itgt);
        if all(~isnan(xy))
          return;
        end
      end
      
      tfsucc = false;
      xy = [];
    end    
    function videoZoom(obj,zoomRadius)
      % Zoom to square window over current frame center with given radius.
      
      [x0,y0] = obj.videoCurrentCenter();
      lims = [x0-zoomRadius,x0+zoomRadius,y0-zoomRadius,y0+zoomRadius];
      axis(obj.gdata.axes_curr,lims);
    end    
    function [xsz,ysz] = videoCurrentSize(obj)
      v = axis(obj.gdata.axes_curr);
      xsz = v(2)-v(1);
      ysz = v(4)-v(3);
    end
    function [x0,y0] = videoCurrentCenter(obj)
      %v = axis(obj.gdata.axes_curr);
      x0 = mean(get(obj.gdata.axes_curr,'XLim'));
      y0 = mean(get(obj.gdata.axes_curr,'YLim'));
    end
    
    function xy = videoClipToVideo(obj,xy)
      % Clip coords to video size.
      %
      % xy (in): [nx2] xy-coords
      %
      % xy (out): [nx2] xy-coords, clipped so that x in [1,nc] and y in [1,nr]
      
      xy = CropInfo.roiClipXY(obj.movieroi,xy);
    end
    function dxdy = videoCurrentUpVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "up" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 
      
      ax = obj.gdata.axes_curr;
      if obj.hasTrx && obj.movieRotateTargetUp
        v = ax.CameraUpVector; % should be norm 1
        dxdy = v(1:2);
      else
        dxdy = [0 1];
      end
    end
    function dxdy = videoCurrentRightVec(obj)
      % The main axis can be rotated, flipped, etc; Get the current unit 
      % "right" vector in (x,y) coords
      %
      % dxdy: [2] unit vector [dx dy] 

      ax = obj.gdata.axes_curr;
      if obj.hasTrx && obj.movieRotateTargetUp
        v = ax.CameraUpVector; % should be norm 1
        parity = mod(strcmp(ax.XDir,'normal') + strcmp(ax.YDir,'normal'),2);
        if parity
          dxdy = [-v(2) v(1)]; % etc
        else
          dxdy = [v(2) -v(1)]; % rotate v by -pi/2.
        end
      else
        dxdy = [1 0];
      end      
    end
    
    function videoPlay(obj)
      obj.videoPlaySegmentCore(obj.currFrame,obj.nframes,...
        'setFrameArgs',{'updateTables',false});
    end
    
    function videoPlaySegment(obj)
      % Play segment centererd at .currFrame
      
      f = obj.currFrame;
      df = obj.moviePlaySegRadius;
      fstart = max(1,f-df);
      fend = min(obj.nframes,f+df);
      obj.videoPlaySegmentCore(fstart,fend,'freset',f,...
        'setFrameArgs',{'updateTables',false,'updateLabels',false});
    end
    
    function videoPlaySegmentCore(obj,fstart,fend,varargin)
      
      [setFrameArgs,freset] = myparse(varargin,...
        'setFrameArgs',{},...
        'freset',nan);
      tfreset = ~isnan(freset);
            
      ticker = tic;
      while true
        % Ways to exit loop:
        % 1. user cancels playback through GUI mutation of gdata.isPlaying
        % 2. fend reached
        % 3. ctrl-c
        
        guidata = obj.gdata;
        if ~guidata.isPlaying
          break;
        end
                  
        dtsec = toc(ticker);
        df = dtsec*obj.moviePlayFPS;
        f = ceil(df)+fstart;
        if f > fend
          break;
        end

        obj.setFrame(f,setFrameArgs{:});
        drawnow('limitrate');

%         dtsec = toc(ticker);
%         pause_time = (f-fstart)/obj.moviePlayFPS - dtsec;
%         if pause_time <= 0,
%           if handles.guidata.mat_lt_8p4
%             drawnow;
%             % MK Aug 2015: There is a drawnow in status update so no need to draw again here
%             % for 2014b onwards.
%             %     else
%             %       drawnow('limitrate');
%           end
%         else
%           pause(pause_time);
%         end
      end
      
      if tfreset
        % AL20170619 passing setFrameArgs a bit fragile; needed for current
        % callers (don't update labels in videoPlaySegment)
        obj.setFrame(freset,setFrameArgs{:}); 
      end
      
      % - icon managed by caller      
    end
    
  end
  
  %% Crop
  methods
    
    function cropSetCropMode(obj,tf)
      if obj.hasTrx && tf
        error('User-specied cropping is unsupported for projects with trx.');
      end
      
      obj.cropCheckCropSizeConsistency();
      if obj.cropIsCropMode,
        obj.syncCropInfoToCurrMov();
      end
      obj.cropIsCropMode = tf;
      obj.notify('cropIsCropModeChanged');
    end
    
    function syncCropInfoToCurrMov(obj)
      
      iMov = obj.currMovie;
      if obj.gtIsGTMode,
        cropInfo = obj.movieFilesAllGTCropInfo{iMov};
      else
        cropInfo = obj.movieFilesAllCropInfo{iMov};
      end
      if ~isempty(cropInfo), 
        for iView=1:obj.nview,
            obj.movieReader(iView).setCropInfo(cropInfo(iView));
        end
      end
    end
    
    function [whDefault,whMaxAllowed] = cropComputeDfltWidthHeight(obj)
      % whDefault: [nview x 2]. cols: width, height
      %   See defns at top of CropInfo.m.
      % whMaxAllowed: [nview x 2]. maximum widthHeight that will fit in all
      %   images, per view
      
      movInfos = [obj.movieInfoAll; obj.movieInfoAllGT];
      nrall = cellfun(@(x)x.info.nr,movInfos); % [(nmov+nmovGT) x nview]
      ncall = cellfun(@(x)x.info.nc,movInfos); % etc
      nrmin = min(nrall,[],1); % [1xnview]
      ncmin = min(ncall,[],1); % [1xnview]
      widthMax = ncmin(:)-1; % col1..col<ncmin>. Minus 1 for posn vs roi 
      heightMax = nrmin(:)-1; % etc
      whMaxAllowed = [widthMax heightMax];
      whDefault = whMaxAllowed/2;
    end
    
    function wh = cropInitCropsAllMovies(obj)
      % Create/initialize default crops for all movies (including GT) in
      % all views
      %
      % wh: [nviewx2] widthHeight of default crop size used. See defns at 
      % top of CropInfo.m.      
      wh = obj.cropComputeDfltWidthHeight;
      obj.cropInitCropsGen(wh,'movieInfoAll','movieFilesAllCropInfo');
      obj.cropInitCropsGen(wh,'movieInfoAllGT','movieFilesAllGTCropInfo');
      obj.preProcNonstandardParamChanged();
      obj.notify('cropCropsChanged');
    end
    function cropInitCropsGen(obj,widthHeight,fldMIA,fldMFACI,varargin)
      % Init crops for certain movies
      % 
      % widthHeight: [nviewx2]
      
      iMov = myparse(varargin,...
        'iMov','__undef__'); % if supplied, indices into .(fldMIA), .(fldMFACI). latter will be initialized

      movInfoAll = obj.(fldMIA);
      [nmov,nvw] = size(movInfoAll);
      szassert(widthHeight,[nvw 2]);

      if strcmp(iMov,'__undef__')
        iMov = 1:nmov;
      else
        iMov = iMov(:)';
      end
      
      for ivw=1:nvw
        whview = widthHeight(ivw,:);
        for i=iMov
          ifo = movInfoAll{i,ivw}.info;
          xyCtr = (1+[ifo.nc ifo.nr])/2;
          posnCtrd = [xyCtr whview];
          obj.(fldMFACI){i}(ivw) = CropInfo.CropInfoCentered(posnCtrd);
        end
      end
    end
      
    function [tfhascrop,roi] = cropGetCropCurrMovie(obj)
      % Current crop per GT mode
      %
      %
      % tfhascrop: scalar logical.
      % roi: [nview x 4]. Applies only when tfhascrop==true; otherwise
      %   indeterminate. roi(ivw,:) is [xlo xhi ylo yhi]. 
      
      iMov = obj.currMovie;
      if iMov==0
        tfhascrop = false;
        roi = [];
      else
        [tfhascrop,roi] = obj.cropGetCropMovieIdx(iMov);
      end      
    end
    
    function [tfhascrop,roi] = cropGetCropMovieIdx(obj,iMov)
      % Current crop per GT mode
      %
      %
      % tfhascrop: scalar logical.
      % roi: [nview x 4]. Applies only when tfhascrop==true; otherwise
      %   indeterminate. roi(ivw,:) is [xlo xhi ylo yhi]. 
      
      PROPS = obj.gtGetSharedProps();
      cropInfo = obj.(PROPS.MFACI){iMov};
      if isempty(cropInfo)
        tfhascrop = false;
        roi = [];
      else
        tfhascrop = true;
        roi = cat(1,cropInfo.roi);
        szassert(roi,[obj.nview 4]);
      end
    end
    
    function cropSetNewRoiCurrMov(obj,iview,roi)
      % Set new crop/roi for current movie (GT-aware).
      %
      % If the crop size has changed, update crops sizes for all movies.
      % 
      % roi: [1x4]. [xlo xhi ylo yhi]. See defns at top of CropInfo.m
      
      iMov = obj.currMovie;
      if iMov==0
        error('No movie selected.');
      end
      
      obj.cropSetNewRoi(iMov,iview,roi);
    end
      
    function cropSetNewRoi(obj,iMov,iview,roi)
      % Set new crop/roi for input movie iMov (GT-aware).
      %
      % If the crop size has changed, update crops sizes for all movies.
      % 
      % roi: [1x4]. [xlo xhi ylo yhi]. See defns at top of CropInfo.m
      % KB added 20200504
      
      movIfo = obj.movieInfoAllGTaware{iMov,iview}.info;
      imnc = movIfo.nc;
      imnr = movIfo.nr;
      tfproper = CropInfo.roiIsProper(roi,imnc,imnr);
      if ~tfproper
        error('ROI extends outside of video. Video image is %d x %d.\n',...
          imnc,imnr);
      end
      
      if ~obj.cropProjHasCrops
        obj.cropInitCropsAllMovies();
        fprintf(1,'Default crop initialized for all movies.\n');
      end
      
      PROPS = obj.gtGetSharedProps();
      roi0 = obj.(PROPS.MFACI){iMov}(iview).roi;
      posn0 = CropInfo.roi2RectPos(roi0);
      posn = CropInfo.roi2RectPos(roi);
      widthHeight0 = posn0(3:4);
      widthHeight = posn(3:4);
      tfSzChanged = ~isequal(widthHeight,widthHeight0);
      tfProceedSet = true;
      if tfSzChanged
        tfOKSz = obj.cropCheckValidCropSize(iview,widthHeight);
        if tfOKSz 
          obj.cropSetSizeAllCrops(iview,widthHeight);
          %warningNoTrace('Crop sizes in all movies altered.');
        else
          warningNoTrace('New roi/crop size is too big for one or more movies in project. ROI not set.');
          tfProceedSet = false;
        end
      end
      if tfProceedSet
        obj.(PROPS.MFACI){iMov}(iview).roi = roi;
      end
      
      % KB 20200113: changing roi for movies with labels will require
      % preproc data to be cleaned out. For now, we are clearing out
      % trackers all together in this case. 
      
      if ~obj.gtIsGTMode && obj.labelPosMovieHasLabels(iMov),
        obj.preProcNonstandardParamChanged();
      else
        % clear CPR cache in any case, even unlabeled movies pose a problem 
        % as tracking frames can be cached
        obj.preProcInitData();
      end

     
%       if ~obj.gtIsGTMode && obj.labelPosMovieHasLabels(iMov),
%         % if this movie has labels, retraining might be necessary
%         % set timestamp for all labels in this movie to now
%         obj.reportLabelChange();
%       end
%       
%       % actually in some codepaths nothing changed, but shouldn't hurt
%       if tfSzChanged,
%         obj.preProcNonstandardParamChanged();
%       end
      obj.notify('cropCropsChanged'); 
    end
    
    function reportLabelChange(obj)
      
      obj.labeledposNeedsSave = true;
      obj.lastLabelChangeTS = now;

    end
    
    function tfOKSz = cropCheckValidCropSize(obj,iview,widthHeight)
      [~,whMaxAllowed] = obj.cropComputeDfltWidthHeight();
      wh = whMaxAllowed(iview,:);
      tfOKSz = all(widthHeight<=wh);
    end
    
    function cropSetSizeAllCrops(obj,iview,widthHeight)
      % Sets crop size for all movies (GT included) for iview. Keeps crops
      % centered as possible
      
      obj.cropSetSizeAllCropsHlp(iview,widthHeight,'movieInfoAll',...
        'movieFilesAllCropInfo');
      obj.cropSetSizeAllCropsHlp(iview,widthHeight,'movieInfoAllGT',...
        'movieFilesAllGTCropInfo');
      obj.preProcNonstandardParamChanged();
      obj.notify('cropCropsChanged'); 
    end
    function cropSetSizeAllCropsHlp(obj,iview,widthHeight,fldMIA,fldMFACI)
      movInfoAll = obj.(fldMIA);
      cropInfos = obj.(fldMFACI);
      for i=1:numel(cropInfos)
        posn = CropInfo.roi2RectPos(cropInfos{i}(iview).roi);
        posn = CropInfo.rectPosResize(posn,widthHeight);
        roi = CropInfo.rectPos2roi(posn);
        
        ifo = movInfoAll{i,iview}.info;
        [roi,tfchanged] = CropInfo.roiSmartFit(roi,ifo.nc,ifo.nr);
        if tfchanged
          warningNoTrace('ROI center moved to stay within movie frame.');
        end
        obj.(fldMFACI){i}(iview).roi = roi;
      end
    end
    
    function cropClearAllCrops(obj)
      % Clear crops for all movies/views, including GT.
      %
      % Note, this is different than cropInitCropsAllMovies. That method
      % creates/sets default crops. This method obliterates all crop infos
      % so that .cropProjHasCrops returns false;
      
      if obj.cropProjHasCrops
        obj.preProcNonstandardParamChanged();
      end
      obj.movieFilesAllCropInfo(:) = {CropInfo.empty(0,0)};
      obj.movieFilesAllGTCropInfo(:) = {CropInfo.empty(0,0)};
      obj.notify('cropCropsChanged'); 
    end
    
    function wh = cropGetCurrentCropWidthHeightOrDefault(obj)
      % If obj.cropProjHasCropInfo is true, get current crop width/height.
      % Otherwise, get default widthHeight.
      %
      % Assumes that the proj has at least one regular movie.
      %
      % wh: [nview x 2]
      
      ci = obj.movieFilesAllCropInfo{1}; 
      if ~isempty(ci)
        roi = cat(1,ci.roi);
        posn = CropInfo.roi2RectPos(roi);
        wh = posn(:,3:4);
      else
        wh = obj.cropComputeDfltWidthHeight();
      end
      szassert(wh,[obj.nview 2]);
    end
    
    function cropCheckCropSizeConsistency(obj)
      % Crop size integrity check. Like a big assert.
      
      if obj.cropProjHasCrops
        cInfoAll = [obj.movieFilesAllCropInfo; obj.movieFilesAllGTCropInfo];
        roisAll = cellfun(@(x)cat(1,x.roi),cInfoAll,'uni',0);
        posnAll = cellfun(@CropInfo.roi2RectPos,roisAll,'uni',0); % posnAll{i} is [nview x 4]
        whAll = cellfun(@(x)x(:,3:4),posnAll,'uni',0);
        whAll = cat(3,whAll{:});
        nCIAll = numel(cInfoAll);
        szassert(whAll,[obj.nview 2 nCIAll]); % ivw, w/h, cropInfo
        assert(isequal(repmat(whAll(:,:,1),1,1,nCIAll),whAll),...
          'Unexpected inconsistency crop sizes.')
      end      
    end
    
    function rois = cropGetAllRois(obj)
      % Get all rois per current GT mode
      %
      % rois: [nmovGTaware x 4 x nview]
      
      if ~obj.cropProjHasCrops
        error('Project does not have crops defined.');
      end
      
      cInfoAll = obj.movieFilesAllCropInfoGTaware;
      rois = cellfun(@(x)cat(1,x.roi),cInfoAll,'uni',0);
      rois = cat(3,rois{:}); % nview, {xlo/xhi/ylo/yhi}, nmov
      rois = permute(rois,[3 2 1]);    
    end 
    
    function hFig = cropMontage(obj,varargin)
      % Create crop montages for all views. Per current GT state.
      %
      % hFig: [nfig] figure handles
      
      [type,imov,nr,nc,plotlabelcolor,figargs] = myparse(varargin,...
        'type','wide',... either 'wide' or 'cropped'. wide shows rois in context of full im. 
        'imov',[],... % show crops for these movs. defaults to 1:nmoviesGTaware
        'nr',9,... % number of rows in montage
        'nc',10,... % etc
        'plotlabelcolor',[1 1 0],...
        'figargs',{'WindowStyle','docked'});
      
      if obj.hasTrx
        error('Unsupported for projects with trx.');
      end
      if ~obj.cropProjHasCrops
        error('Project does not have crops defined.');
      end
      
      if isempty(imov)
        imov = 1:obj.nmoviesGTaware;
      end
      
      switch lower(type)
        case 'wide', tfWide = true;
        case 'cropped', tfWide = false;
        otherwise, assert(false);
      end
      
      % get MFTable to pull first frame of each mov
      mov = obj.movieFilesAllFullGTaware(imov,:);
      nmov = size(mov,1);
      if nmov==0
        error('No movies.');
      end
      frm = ones(nmov,1);
      iTgt = ones(nmov,1);
      tblMFT = table(mov,frm,iTgt);
      wbObj = WaitBarWithCancel('Montage');
      oc = onCleanup(@()delete(wbObj));      
      I1 = CPRData.getFrames(tblMFT,...
        'movieInvert',obj.movieInvert,...
        'wbObj',wbObj);

      roisAll = obj.cropGetAllRois; 
      roisAll = roisAll(imov,:,:);
      
      nvw = obj.nview;
      if ~tfWide
        for iimov=1:nmov
          for ivw=1:nvw
            roi = roisAll(iimov,:,ivw);
            I1{iimov,ivw} = I1{iimov,ivw}(roi(3):roi(4),roi(1):roi(2));
          end
        end
      end

      nplotperbatch = nr*nc;
      nbatch = ceil(nmov/nplotperbatch);
      szassert(I1,[nmov nvw]);
      szassert(roisAll,[nmov 4 nvw]);
      hFig = gobjects(0,1);
      for ivw=1:nvw
        roi1 = roisAll(1,:,ivw);
        pos1 = CropInfo.roi2RectPos(roi1);
        wh = pos1(3:4)+1;
        
        if tfWide
          imsz = cellfun(@size,I1(:,ivw),'uni',0);
          imsz = cat(1,imsz{:}); % will err if some ims are color etc
          imszUn = unique(imsz,'rows');
          tfImsHeterogeneousSz = size(imszUn,1)>1;
        end
        
        for ibatch=1:nbatch
          iimovs = (1:nplotperbatch) + (ibatch-1)*nplotperbatch;
          iimovs(iimovs>nmov) = [];
          figstr = sprintf('movs %d->%d. view %d.',...
            iimovs(1),iimovs(end),ivw);
          titlestr = sprintf('movs %d->%d. view %d. [w h]: %s',...
            iimovs(1),iimovs(end),ivw,mat2str(wh));
          
          hFig(end+1,1) = figure(figargs{:}); %#ok<AGROW>
          hFig(end).Name = figstr;
          
          if tfWide
            Shape.montage(I1(:,ivw),nan(nmov,2),...
              'fig',hFig(end),...
              'nr',nr,'nc',nc,'idxs',iimovs,...
              'rois',roisAll(:,:,ivw),...
              'imsHeterogeneousSz',tfImsHeterogeneousSz,...
              'framelbls',arrayfun(@num2str,imov(iimovs),'uni',0),...
              'framelblscolor',plotlabelcolor,...
              'titlestr',titlestr);
          else
            Shape.montage(I1(:,ivw),nan(nmov,2),...
              'fig',hFig(end),...
              'nr',nr,'nc',nc,'idxs',iimovs,...
              'framelbls',arrayfun(@num2str,imov(iimovs),'uni',0),...
              'framelblscolor',plotlabelcolor,...
              'titlestr',titlestr);
          end
        end
      end        
    end
            
  end
 
  
  %% Navigation
  methods
    
    function navPrefsUI(obj)
      NavPrefs(obj);
    end
    
    function setMFT(obj,iMov,frm,iTgt)
      if isa(iMov,'MovieIndex')
        if obj.currMovIdx~=iMov
          obj.movieSetMIdx(iMov);
        end
      else
        if obj.currMovie~=iMov
          obj.movieSet(iMov);
        end
      end
      obj.setFrameAndTarget(frm,iTgt);
    end
  
    function tfSetOccurred = setFrameProtected(obj,frm,varargin)
      % Protected set against frm being out-of-bounds for current target.
      
      if obj.hasTrx 
        iTgt = obj.currTarget;
        if ~obj.frm2trx(frm,iTgt)
          tfSetOccurred = false;
          return;
        end
      end
      
      tfSetOccurred = true;
      obj.setFrame(frm,varargin{:});      
    end
    
    function setFrame(obj,frm,varargin)
      % Set movie frame, maintaining current movie/target.
      %
      % CTRL-C note: This is fairly ctrl-c safe; a ctrl-c break may leave
      % obj state a little askew but it should be cosmetic and another
      % (full/completed) setFrame() call should fix things up. We could
      % prob make it even more Ctrl-C safe with onCleanup-plus-a-flag.
      
      setframetic = tic;
      starttime = setframetic;
            
      [tfforcereadmovie,tfforcelabelupdate,updateLabels,updateTables,...
        updateTrajs,changeTgtsIfNec] = myparse(varargin,...
        'tfforcereadmovie',false,...
        'tfforcelabelupdate',false,...
        'updateLabels',true,...
        'updateTables',true,...
        'updateTrajs',true,...
        'changeTgtsIfNec',false... % if true, will alter the current target if it is not live in frm
        );
            
      %fprintf('setFrame %d, parse inputs took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      
      if obj.hasTrx
        assert(~obj.isMultiView,'MultiView labeling not supported with trx.');
        frm2trxThisFrm = obj.frm2trx(frm,:);
        iTgt = obj.currTarget;
        if ~frm2trxThisFrm(1,iTgt)
          if changeTgtsIfNec
            iTgtsLive = find(frm2trxThisFrm);
            if isempty(iTgtsLive)
              error('Labeler:target','No targets live in frame %d.',frm);              
            else
              iTgtsLiveDist = abs(iTgtsLive-iTgt);
              itmp = argmin(iTgtsLiveDist);
              iTgtNew = iTgtsLive(itmp);
              warningNoTrace('Target %d is not live in frame %d. Changing to target %d.\n',...
                iTgt,frm,iTgtNew);
              obj.setFrameAndTarget(frm,iTgtNew);
              return;
            end
          else
            error('Labeler:target','Target %d not live in frame %d.',...
              iTgt,frm);
          end
        end
      end
      
      %fprintf('setFrame %d, trx stuff took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      
      % Remainder nearly identical to setFrameAndTarget()
      try
        obj.hlpSetCurrPrevFrame(frm,tfforcereadmovie);
      catch ME
        warning('Could not set previous frame: %s',ME.message);
      end
      
      %fprintf('setFrame %d, setcurrprevframe took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      
      if obj.hasTrx && obj.movieCenterOnTarget && ~obj.movieCenterOnTargetLandmark
        assert(~obj.isMultiView);
        obj.videoCenterOnCurrTarget();
      elseif obj.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTargetPoint();
      end
      
      %fprintf('setFrame %d, center and rotate took %f seconds\n',frm,toc(setframetic)); setframetic = tic;

      
      if updateLabels
        obj.labelsUpdateNewFrame(tfforcelabelupdate);
      end
      
      %fprintf('setFrame %d, updatelabels took %f seconds\n',frm,toc(setframetic)); setframetic = tic;
      
      if updateTables
        obj.updateTrxTable();
%         obj.updateCurrSusp();
      end
      
      %fprintf('setFrame %d, update tables took %f seconds\n',frm,toc(setframetic)); setframetic = tic;

      
      if updateTrajs
        obj.updateTrx();
      end
      
      %fprintf('setFrame %d, update showtrx took %f seconds\n',frm,toc(setframetic));

      
      %fprintf('setFrame to %d took %f seconds\n',frm,toc(starttime));
      
    end
    
%     function setTargetID(obj,tgtID)
%       % Set target ID, maintaining current movie/frame.
%       
%       iTgt = obj.trxIdPlusPlus2Idx(tgtID+1);
%       assert(~isnan(iTgt),'Invalid target ID: %d.');
%       obj.setTarget(iTgt);
%     end

    function clickTarget(obj,h,evt,iTgt)
      
      if strcmpi(obj.gdata.figure.SelectionType,'open'),
        obj.SetStatus(sprintf('Switching to target %d...',iTgt));
        %fprintf('Switching to target %d\n',iTgt);
        obj.setTarget(iTgt);
        obj.ClearStatus();
      end
      
    end
    
    function setTarget(obj,iTgt)
      % Set target index, maintaining current movie/frameframe.
      % iTgt: INDEX into obj.trx
      
      validateattributes(iTgt,{'numeric'},...
        {'positive' 'integer' '<=' obj.nTargets});
      
      frm = obj.currFrame;
      if ~obj.frm2trx(frm,iTgt)
        error('Labeler:target',...
          'Target idx %d is not live at current frame (%d).',iTgt,frm);
      end
      
      prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      if obj.hasTrx 
        obj.labelsUpdateNewTarget(prevTarget);
        if obj.movieCenterOnTarget && ~obj.movieCenterOnTargetLandmark
          obj.videoCenterOnCurrTarget();
        elseif obj.movieCenterOnTargetLandmark
          obj.videoCenterOnCurrTargetPoint();
        end
      end
%       obj.updateCurrSusp();
      obj.updateShowTrx();
    end
        
    function setFrameAndTarget(obj,frm,iTgt)
      % Set to new frame and target for current movie.
      % Prefer setFrame() or setTarget() if possible to
      % provide better continuity wrt labeling etc.
     
      validateattributes(iTgt,{'numeric'},...
        {'positive' 'integer' '<=' obj.nTargets});

      if ~obj.isinit && obj.hasTrx && ~obj.frm2trx(frm,iTgt)
        error('Labeler:target',...
          'Target idx %d is not live at current frame (%d).',iTgt,frm);
      end

      % 2nd arg true to match legacy
      try
        obj.hlpSetCurrPrevFrame(frm,true);
      catch ME
        warning('Could not set previous frame: %s', ME.message);
      end
      
      prevTarget = obj.currTarget;
      obj.currTarget = iTgt;
      
      if obj.hasTrx && obj.movieCenterOnTarget && ~obj.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTarget();
        obj.videoZoom(obj.targetZoomRadiusDefault);
      elseif obj.movieCenterOnTargetLandmark
        obj.videoCenterOnCurrTargetPoint();
      end
      if ~obj.isinit
        obj.labelsUpdateNewFrameAndTarget(obj.prevFrame,prevTarget);
        obj.updateTrxTable();
%         obj.updateCurrSusp();
        obj.updateShowTrx();
      end
    end    
    
    function tfSetOccurred = frameUpDF(obj,df)
      f = min(obj.currFrame+df,obj.nframes);
      tfSetOccurred = obj.setFrameProtected(f); 
    end
    
    function tfSetOccurred = frameDownDF(obj,df)
      f = max(obj.currFrame-df,1);
      tfSetOccurred = obj.setFrameProtected(f);
    end
    
    function tfSetOccurred = frameUp(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      tfSetOccurred = obj.frameUpDF(df);
    end
  end
  methods (Static) % seek utils
    function [tffound,f] = seekBigLpos(lpos,f0,df,iTgt)
      % lpos: [npts x d x nfrm x ntgt]
      % f0: starting frame
      % df: frame increment
      % iTgt: target of interest
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label, applicable if
      %   tffound==true
      
      if isempty(lpos),
        tffound = false;
        f = f0;
        return;
      end
      
      [npts,d,nfrm,ntgt] = size(lpos); %#ok<ASGLU>
      assert(d==2);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt = 1:npts
          %for j = 1:2
          if ~isnan(lpos(ipt,1,f,iTgt))
            tffound = true;
            return;
          end
          %end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end
    function [tffound,f] = seekSmallLpos(lpos,f0,df)
      % lpos: [npts x nfrm]
      % f0: starting frame
      % df: frame increment
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label, applicable if
      %   tffound==true
      
      [npts,nfrm] = size(lpos);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt=1:npts
          if ~isnan(lpos(ipt,f))
            tffound = true;
            return;
          end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end
    function [tffound,f] = seekSmallLposThresh(lpos,f0,df,th,cmp)
      % lpos: [npts x nfrm]
      % f0: starting frame
      % df: frame increment
      % th: threshold
      % cmp: comparitor
      % 
      % tffound: logical
      % f: first frame encountered with (non-nan) label that satisfies 
      % comparison with threshold, applicable if tffound==true
      
      switch cmp
        case '<',  cmp = @lt;
        case '<=', cmp = @le;
        case '>',  cmp = @gt;
        case '>=', cmp = @ge;
      end
          
      [npts,nfrm] = size(lpos);
      
      f = f0+df;
      while 0<f && f<=nfrm
        for ipt=1:npts
          if cmp(lpos(ipt,f),th)
            tffound = true;
            return;
          end
        end
        f = f+df;
      end
      tffound = false;
      f = nan;
    end
  end
  methods
    function tfSetOccurred = frameDown(obj,tfBigstep)
      if tfBigstep
        df = obj.movieFrameStepBig;
      else
        df = 1;
      end
      tfSetOccurred = obj.frameDownDF(df);
    end
    
    function frameUpNextLbled(obj,tfback,varargin)
      % call obj.setFrame() on next labeled frame. 
      % 
      % tfback: optional. if true, seek backwards.
            
      if ~obj.hasMovie || obj.currMovie==0
        return;
      end
      
      lpos = myparse(varargin,...
        'lpos','__UNSET__'); % optional, provide "big" lpos array to use instead of .labeledposCurrMovie
      
      if tfback
        df = -1;
      else
        df = 1;
      end
      
      if strcmp(lpos,'__UNSET__')
        lpos = obj.labeledposCurrMovie;
      elseif isempty(lpos)
        % edge case
        return;
      else
        szassert(lpos,[obj.nLabelPoints 2 obj.nframes obj.nTargets]);
      end
        
      [tffound,f] = Labeler.seekBigLpos(lpos,obj.currFrame,df,...
        obj.currTarget);
      if tffound
        obj.setFrameProtected(f);
      end
    end
    
    function [x,y,th] = currentTargetLoc(obj,varargin)
      % Return current target loc, or movie center if no target
      
      nowarn = myparse(varargin,...
        'nowarn',false ... % If true, don't warn about incompatible .currFrame/.currTrx. 
          ... % The incompatibility really shouldn't happen but it's an edge case
          ... % and not worth the trouble to fix for now.
        );
      
      assert(~obj.isMultiView,'Not supported for MultiView.');
      
      if obj.hasTrx
        cfrm = obj.currFrame;
        ctrx = obj.currTrx;


        if cfrm < ctrx.firstframe || cfrm > ctrx.endframe
          if ~nowarn
            warningNoTrace('Labeler:target',...
              'No track for current target at frame %d.',cfrm);
          end
          movroictr = obj.movieroictr; % [1x2]
          x = round(movroictr(1));
          y = round(movroictr(2));
          th = 0;
        else
          i = cfrm - ctrx.firstframe + 1;
          x = ctrx.x(i);
          y = ctrx.y(i);
          th = ctrx.theta(i);
        end
      else
        movroictr = obj.movieroictr; % [1x2]
        x = round(movroictr(1));
        y = round(movroictr(2));
        th = 0;
      end
    end
    
    function [x,y,th] = targetLoc(obj,iMov,iTgt,frm)
      % Added by KB 20181010
      % Return current target center and orientation for input movie, target, and frame
      
      assert(obj.hasTrx,'This should not be called when videos do not have trajectories');
      
      if iMov == obj.currMovie && iTgt == obj.currTarget,
        ctrx = obj.currTrx;
      else
        trxfname = obj.trxFilesAllFullGTaware{iMov,1};
        movIfo = obj.movieInfoAllGTaware{iMov};
        [s.trx,s.frm2trx] = obj.getTrx(trxfname,movIfo.nframes);
        ctrx = s.trx(iTgt);
      end
      cfrm = frm;

      assert(cfrm >= ctrx.firstframe && cfrm <= ctrx.endframe);
      i = cfrm - ctrx.firstframe + 1;
      x = double(ctrx.x(i));
      y = double(ctrx.y(i));
      th = double(ctrx.theta(i));
    end
    function setSelectedFrames(obj,frms)
      if isempty(frms)
        obj.selectedFrames = frms;        
      elseif ~obj.hasMovie
        error('Labeler:noMovie',...
          'Cannot set selected frames when no movie is loaded.');
      else
        validateattributes(frms,{'numeric'},{'integer' 'vector' '>=' 1 '<=' obj.nframes});
        obj.selectedFrames = frms;  
      end
    end
                
    function updateTrxTable(obj)
      % based on .frm2trxm, .currFrame, .labeledpos
      
      %starttime = tic;
      tbl = obj.gdata.tblTrx;
      if ~obj.hasTrx || ~obj.hasMovie || obj.currMovie==0 % Can occur during movieSet(), when invariants momentarily broken
        ischange = ~isempty(obj.tblTrxData);
        if ischange,
          obj.tblTrxData = zeros(0,2);
          set(tbl,'Data',cell(0,2));
        end
        %fprintf('Time in updateTrxTable: %f\n',toc(starttime));
        return;
      end
      
      f = obj.currFrame;
      tfLive = obj.frm2trx(f,:);
      idxLive = find(tfLive);
      idxLive = idxLive(:);
      lpos = obj.labeledposCurrMovie;
      tfLbled = arrayfun(@(x)any(lpos(:,1,f,x)),idxLive); % nans counted as 0
      ischange = true;
      tblTrxData = [idxLive,tfLbled]; %#ok<*PROP>
      if ~isempty(obj.tblTrxData),
        ischange = ndims(tblTrxData) ~= ndims(obj.tblTrxData) || ...
          any(size(tblTrxData) ~= size(obj.tblTrxData)) || ...
          any(tblTrxData(:) ~= obj.tblTrxData(:));
      end
      if ischange,
        
%         [nrold,ncold] = size(obj.tblTrxData);
%         [nrnew,ncnew] = size(tblTrxData);
%         ischange = true([nrnew,ncnew]);
%         nr = min(nrold,nrnew);
%         nc = min(ncold,ncnew);
%         ischange(1:nr,1:nc) = obj.tblTrxData(1:nr,1:nc) ~= tblTrxData(1:nr,1:nc);
%         [is,js] = find(ischange);
        
        obj.tblTrxData = tblTrxData;
        tbldat = [num2cell(idxLive) num2cell(tfLbled)];
        %tbl.setDataFast(is,js,tbldat(ischange),nrnew,ncnew);
        %tbl.setDataUnsafe(tbldat);
        set(tbl,'Data',tbldat);
      end

      %fprintf('Time in updateTrxTable: %f\n',toc(starttime));
    end
    
    % TODO: Move this into UI
    function updateFrameTableIncremental(obj)
      % assumes .labelpos and tblFrames differ at .currFrame at most
      %
      % might be unnecessary/premature optim
      
      tbl = obj.gdata.tblFrames;
      dat = get(tbl,'Data');
      tblFrms = cell2mat(dat(:,1));      
      cfrm = obj.currFrame;
      tfRow = (tblFrms==cfrm);
      
      [nTgtsCurFrm,nPtsCurFrm] = obj.labelPosLabeledFramesStats(cfrm);
      if nTgtsCurFrm>0
        if any(tfRow)
          assert(nnz(tfRow)==1);
          iRow = find(tfRow);
          dat(iRow,2:3) = {nTgtsCurFrm nPtsCurFrm};
          
          set(tbl,'Data',dat);
          %tbl.setDataFast([iRow iRow],2:3,{nTgtsCurFrm nPtsCurFrm},...
          %  size(dat,1),size(dat,2));
        else          
          dat(end+1,:) = {cfrm nTgtsCurFrm nPtsCurFrm};
          n = size(dat,1);
          tblFrms(end+1,1) = cfrm;
          [~,idx] = sort(tblFrms);
          dat = dat(idx,:);
          iRow = find(idx==n);
          
          set(tbl,'Data',dat);
        end
      else
        iRow = [];
        if any(tfRow)
          assert(nnz(tfRow)==1);
          dat(tfRow,:) = [];
          set(tbl,'Data',dat);
        end
      end
      
      %tbl.SelectedRows = iRow;

      nTgtsTot = sum(cell2mat(dat(:,2)));

      % dat should equal get(tbl,'Data')
      if obj.hasMovie
        PROPS = obj.gtGetSharedProps();
        %obj.gdata.labelTLInfo.setLabelsFrame();
        obj.(PROPS.MFAHL)(obj.currMovie) = nTgtsTot;
      end
      
      tx = obj.gdata.txTotalFramesLabeled;
      tx.String = num2str(nTgtsTot);
    end    
    function updateFrameTableComplete(obj)
      [nTgts,nPts] = obj.labelPosLabeledFramesStats();
      assert(isequal(nTgts>0,nPts>0));
      tfFrm = nTgts>0;
      iFrm = find(tfFrm);

      nTgtsLbledFrms = nTgts(tfFrm);
      dat = [num2cell(iFrm) num2cell(nTgtsLbledFrms) num2cell(nPts(tfFrm)) ];
      tbl = obj.gdata.tblFrames;
      set(tbl,'Data',dat);

      nTgtsTot = sum(nTgtsLbledFrms);

      if obj.hasMovie
        PROPS = obj.gtGetSharedProps();
        %obj.gdata.labelTLInfo.setLabelsFrame(1:obj.nframes);
        obj.(PROPS.MFAHL)(obj.currMovie) = nTgtsTot;
      end
      
      tx = obj.gdata.txTotalFramesLabeled;
      tx.String = num2str(nTgtsTot);
    end
  end
  
  methods (Hidden)

    function hlpSetCurrPrevFrame(obj,frm,tfforce)
      % helper for setFrame, setFrameAndTarget

      %ticinfo = tic;
      gd = obj.gdata;

      currFrmOrig = obj.currFrame;      
      imcurr = gd.image_curr;
      currImOrig = struct('CData',imcurr.CData,...
          'XData',imcurr.XData,'YData',imcurr.YData);
        
      %fprintf('hlpSetCurrPrevFrame 1: %f\n',toc(ticinfo)); ticinfo = tic;  
      nfrs = min([obj.movieReader(:).nframes]);
      if frm>nfrs
        s1 = sprintf('Cannot browse to frame %d because the maximum',frm);
        s2 = sprintf('number of frames reported by matlab is %d.',nfrs);
        s3 = sprintf('Browsing to frame %d.',nfrs);
        s4 = sprintf('The number of frames reported in a video can differ');
        s5 = sprintf('based on the codecs installed. We tried to make');
        s6 = sprintf('them consistent but there are no solutions other');
        s7 = sprintf('encoding the video using a simpler codec.');

        warndlg({s1,s2,s3,s4,s5,s6,s7}, 'Frame out of limit');
        frm = nfrs;
      end
      if obj.currFrame~=frm || tfforce
        imsall = gd.images_all;
        tfCropMode = obj.cropIsCropMode;        
        for iView=1:obj.nview
          if tfCropMode
            [obj.currIm{iView},~,currImRoi] = ...
              obj.movieReader(iView).readframe(frm,...
              'doBGsub',obj.movieViewBGsubbed,'docrop',false);                   
          else
            [obj.currIm{iView},~,currImRoi] = ...
              obj.movieReader(iView).readframe(frm,...
              'doBGsub',obj.movieViewBGsubbed,'docrop',true);                  
          end          
          %fprintf('hlpSetCurrPrevFrame 2: %f\n',toc(ticinfo)); ticinfo = tic;  
          set(imsall(iView),...
            'CData',obj.currIm{iView},...
            'XData',currImRoi(1:2),...
            'YData',currImRoi(3:4));
          %fprintf('hlpSetCurrPrevFrame 3: %f\n',toc(ticinfo)); ticinfo = tic;  
        end
        obj.gdata.labelTLInfo.newFrame(frm);
        %fprintf('hlpSetCurrPrevFrame 4: %f\n',toc(ticinfo)); ticinfo = tic;  
        obj.currFrame = frm;
        %fprintf('hlpSetCurrPrevFrame 4a: %f\n',toc(ticinfo)); ticinfo = tic;
        set(obj.gdata.edit_frame,'String',num2str(frm));
        sldval = (frm-1)/(obj.nframes-1);
        if isnan(sldval)
          sldval = 0;
        end
        set(obj.gdata.slider_frame,'Value',sldval);
        if ~obj.isinit
          hlpGTUpdateAxHilite(obj);
        end
        
        if obj.gtIsGTMode
          GTManager('cbkCurrMovFrmTgtChanged',obj.gdata.GTMgr);
        end
        
        if ~isempty(obj.tracker),
          obj.tracker.newLabelerFrame();
        end
          
        %fprintf('hlpSetCurrPrevFrame 5: %f\n',toc(ticinfo)); ticinfo = tic;
      end
      
      % AL20180619 .currIm is probably an unnec prop
      obj.prevFrame = currFrmOrig;
      currIm1Nr = size(obj.currIm{1},1);
      currIm1Nc = size(obj.currIm{1},2);
      if ~isequal([size(currImOrig.CData,1) size(currImOrig.CData,2)],...
          [currIm1Nr currIm1Nc])
        % In this scenario we do not use currIm1Orig b/c axes_prev and 
        % axes_curr are linked and that can force the axes into 'manual'
        % XLimMode and so on. Generally it is disruptive to view-handling.
        % Ideally maybe we would prefer to catch/handle this in view code,
        % but there is no convenient hook between setting the image CData
        % and display.
        %
        % Maybe more importantly, the only time the sizes will disagree are 
        % in edge cases eg when a project is loaded or changed. In this 
        % case currIm1Orig is not going to represent anything meaningful 
        % anyway.
        obj.prevIm = struct('CData',zeros(currIm1Nr,currIm1Nc),...
          'XData',1:currIm1Nc,'YData',1:currIm1Nr);
      else
        obj.prevIm = currImOrig;
      end
      %fprintf('hlpSetCurrPrevFrame 6: %f\n',toc(ticinfo)); ticinfo = tic;  
      obj.prevAxesImFrmUpdate(tfforce);
      %fprintf('hlpSetCurrPrevFrame 7: %f\n',toc(ticinfo));
      %fprintf('hlpSetCurrPrevFrame: %f\n',toc(ticinfo));
      
    end
    
    function hlpGTUpdateAxHilite(obj)
      if obj.gtIsGTMode
        tfHilite = obj.gtCurrMovFrmTgtIsInGTSuggestions();
      else
        tfHilite = false;
      end
      obj.gdata.allAxHiliteMgr.setHighlight(tfHilite);
    end
  end
  
  %% PrevAxes
  methods 
    
    function setPrevAxesMode(obj,pamode,pamodeinfo)
      % Set .prevAxesMode, .prevAxesModeInfo
      %
      % pamode: PrevAxesMode
      % pamodeinfo: (optional) userdata for pamode.
      
      if exist('pamodeinfo','var')==0
        pamodeinfo = [];
      end
      contents = cellstr(get(obj.gdata.popupmenu_prevmode,'String'));
      v1 = get(obj.gdata.popupmenu_prevmode,'Value');
      switch pamode
        case PrevAxesMode.FROZEN,
          v2 = find(strcmpi(contents,'Reference'));
        case PrevAxesMode.LASTSEEN,
          v2 = find(strcmpi(contents,'Previous frame'));
        otherwise
          error('Unknown previous axes mode');
      end
      if v2 ~= v1,
        set(obj.gdata.popupmenu_prevmode,'Value',v2);
      end

      obj.prevAxesMode = pamode;
      
      switch pamode
        case PrevAxesMode.LASTSEEN
          %obj.prevAxesModeInfo = pamodeinfo;
          obj.prevAxesImFrmUpdate();
          obj.prevAxesLabelsUpdate();
          gd = obj.gdata;
          axp = gd.axes_prev;
          set(axp,...
            'CameraUpVectorMode','auto',...
            'CameraViewAngleMode','auto');
          gd.hLinkPrevCurr.Enabled = 'on'; % links X/Ylim, X/YDir
          gd.pushbutton_freezetemplate.Visible = 'off';
        case PrevAxesMode.FROZEN
          obj.prevAxesFreeze(pamodeinfo);
          obj.gdata.pushbutton_freezetemplate.Visible = 'on';
        otherwise
          assert(false);
      end
    end
    
    function prevAxesFreeze(obj,freezeInfo)
      % Freeze the current frame/labels in the previous axis. Sets
      % .prevAxesMode, .prevAxesModeInfo.
      %
      % freezeInfo: Optional freezeInfo to apply. If not supplied,
      % image/labels taken from current movie/frame/etc.
      
      if ~obj.hasMovie,
        return;
      end
      
      set(obj.gdata.popupmenu_prevmode,'Visible','on');
      set(obj.gdata.pushbutton_freezetemplate,'Visible','on');
      gd = obj.gdata;
      if isequal(freezeInfo,[])
        axc = gd.axes_curr;
        freezeInfo = struct(...
          'iMov',obj.currMovie,...
          'frm',obj.currFrame,...
          'iTgt',obj.currTarget,...
          'im',obj.currIm{1},...
          'isrotated',false);
        if isfield(obj.prevAxesModeInfo,'dxlim'),
          freezeInfo.dxlim = obj.prevAxesModeInfo.dxlim;
          freezeInfo.dylim = obj.prevAxesModeInfo.dylim;
        end
        freezeInfo = obj.SetPrevMovieInfo(freezeInfo);
        freezeInfo = obj.GetDefaultPrevAxes(freezeInfo);
      end
      
      success = true;
      if ~obj.isPrevAxesModeInfoSet(freezeInfo),
        [success,freezeInfo] = obj.FixPrevModeInfo(PrevAxesMode.FROZEN,freezeInfo);
      end
      if ~success,
        freezeInfo.iMov = [];
        freezeInfo.frm = [];
        freezeInfo.iTgt = [];
        freezeInfo.im = [];
        freezeInfo.isrotated = false;
        gd.image_prev.CData = 0;
        gd.txPrevIm.String = '';
      else
        gd.image_prev.XData = freezeInfo.xdata;
        gd.image_prev.YData = freezeInfo.ydata;
        gd.image_prev.CData = freezeInfo.im;
        gd.txPrevIm.String = sprintf('Frame %d',freezeInfo.frm);
        if obj.hasTrx,
          gd.txPrevIm.String = [gd.txPrevIm.String,sprintf(', Target %d',freezeInfo.iTgt)];
        end
        gd.txPrevIm.String = [gd.txPrevIm.String,sprintf(', Movie %d',freezeInfo.iMov)];
      end
      obj.prevAxesSetLabels(freezeInfo.iMov,freezeInfo.frm,freezeInfo.iTgt,freezeInfo);
      
      gd.hLinkPrevCurr.Enabled = 'off';
      axp = gd.axes_prev;
      axcProps = freezeInfo.axes_curr;
      for prop=fieldnames(axcProps)',prop=prop{1}; %#ok<FXSET>
        axp.(prop) = axcProps.(prop);
      end
      if freezeInfo.isrotated,
        axp.CameraUpVectorMode = 'auto';
      end
      % Setting XLim/XDir etc unnec coming from PrevAxesMode.LASTSEEN, but 
      % sometimes nec eg for a "refreeze"
      
      obj.prevAxesMode = PrevAxesMode.FROZEN;
      obj.prevAxesModeInfo = freezeInfo;
    end
    
    function prevAxesImFrmUpdate(obj,tfforce)

      if nargin < 2,
        tfforce = false;
      end
      
      if ~obj.hasMovie || isempty(obj.prevAxesMode),
        return;
      end
      
      set(obj.gdata.popupmenu_prevmode,'Visible','on');
      % update prevaxes image and txframe based on .prevIm, .prevFrame
      switch obj.prevAxesMode
        case PrevAxesMode.LASTSEEN
          gd = obj.gdata;
          set(gd.image_prev,obj.prevIm);
          gd.txPrevIm.String = sprintf('Frame: %d',obj.prevFrame);
          if obj.hasTrx,
            gd.txPrevIm.String = [gd.txPrevIm.String,sprintf(', Target %d',obj.currTarget)];
          end
        case PrevAxesMode.FROZEN,          
          if tfforce && obj.isPrevAxesModeInfoSet(),
            obj.prevAxesModeInfo = obj.SetPrevMovieInfo(obj.prevAxesModeInfo);
            obj.prevAxesFreeze(obj.prevAxesModeInfo);
          end
      end
    end

    function isvalid = isPrevAxesModeInfoSet(obj,ModeInfo)
      
      if nargin < 2,
        ModeInfo = obj.prevAxesModeInfo;
      end
      isvalid = ~isempty(ModeInfo) && isstruct(ModeInfo) && isfield(ModeInfo,'frm') ...
        && ~isempty(ModeInfo.frm) && ModeInfo.frm > 0 && ...
        isfield(ModeInfo,'iMov') && ~isempty(ModeInfo.iMov);
      
    end
    
    function clearPrevAxesModeInfo(obj)
      
      obj.prevAxesModeInfo.iMov = [];
      obj.prevAxesModeInfo.iTgt = [];
      obj.prevAxesModeInfo.frm = [];
      
      obj.prevAxesModeInfo.im = [];
      obj.prevAxesModeInfo.isrotated = false;
      if isfield(obj.gdata,'image_prev') && ishandle(obj.gdata.image_prev),
        obj.gdata.image_prev.CData = 0;
      end
      if isfield(obj.gdata,'txPrevIm') && ishandle(obj.gdata.txPrevIm),
        obj.gdata.txPrevIm.String = '';
      end
      
    end
    
    function islabeled = currFrameIsLabeled(obj)
      
      lpos = obj.labeledposGTaware;
      lpos = lpos{obj.currMovie}(:,:,obj.currFrame,obj.currTarget);
      islabeled = all(~isnan(lpos(:)));
    end
    
    function prevAxesLabelsUpdate(obj)
      % Update (if required) .lblPrev_ptsH, .lblPrev_ptsTxtH based on 
      % .prevFrame etc 
      % KB HERE
      if obj.isinit || ~obj.hasMovie,
        return;
      end
      
      islabeled = obj.currFrameIsLabeled();
      if islabeled,
        set(obj.gdata.pushbutton_freezetemplate,'Enable','on');
      else
        set(obj.gdata.pushbutton_freezetemplate,'Enable','off');
      end
      
      if obj.prevAxesMode==PrevAxesMode.FROZEN,
        return;
      end
      if ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        obj.prevAxesSetLabels(obj.currMovie,obj.prevFrame,obj.currTarget);
      else
        LabelCore.setPtsOffaxis(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      end
    end
    
    function prevAxesLabelsRedraw(obj)
      % Maybe should be an option to prevAxesLabelsUpdate()
      %

      if obj.prevAxesMode==PrevAxesMode.FROZEN
        % Strictly speaking this could lead to an unexpected change in
        % frozen reference frame if the underlying labels for that frame
        % have changed
        
        freezeInfo = obj.prevAxesModeInfo;
        obj.prevAxesSetLabels(freezeInfo.iMov,freezeInfo.frm,freezeInfo.iTgt,freezeInfo);
      elseif ~isnan(obj.prevFrame) && ~isempty(obj.lblPrev_ptsH)
        obj.prevAxesSetLabels(obj.currMovie,obj.prevFrame,obj.currTarget);
      else
        LabelCore.setPtsOffaxis(obj.lblPrev_ptsH,obj.lblPrev_ptsTxtH);
      end
    end
    
    function [success,paModeInfo] = FixPrevModeInfo(obj,paMode,paModeInfo)
      
      if nargin < 2,
        paMode = obj.prevAxesMode;
        paModeInfo = obj.prevAxesModeInfo;
      end
      % KB 20181010 - make sure the frozen frame is labeled
      if paMode~=PrevAxesMode.FROZEN,
        success = true;
        return;
      end
        
      % make sure the previous frame is labeled
      success = false;
      lpos = obj.labeledposGTaware;
      if obj.isPrevAxesModeInfoSet(paModeInfo),
        if numel(lpos) >= paModeInfo.iMov,
          if isfield(paModeInfo,'iTgt'),
            iTgt = paModeInfo.iTgt;
          else
            iTgt = 1;
          end
          if size(lpos{paModeInfo.iMov},4) >= iTgt && ...
              size(lpos{paModeInfo.iMov},3) >= paModeInfo.frm,
            success = all(all(all(~isnan(lpos{paModeInfo.iMov}(:,:,paModeInfo.frm,iTgt,:)))));
          end
        end
        if success,
          return;
        end
      end
      % find first labeled frame
      %frm = paModeInfo.frm;
      %iMov = paModeInfo.iMov;
      %iTgt = paModeInfo.iTgt;
      
      if ~isfield(paModeInfo,'axes_curr'),
        paModeInfo = obj.SetPrevAxesProperties(paModeInfo);
      end
      
      [tffound,iMov,frm,iTgt] = obj.labelFindOneLabeledFrameEarliest();
      if ~tffound,
        paModeInfo.frm = [];
        paModeInfo.iTgt = [];
        paModeInfo.iMov = [];
        if nargin < 2,
          obj.prevAxesModeInfo = paModeInfo;
        end
        return;
      end
      paModeInfo.frm = frm;
      paModeInfo.iTgt = iTgt;
      paModeInfo.iMov = iMov;
      
      paModeInfo = obj.SetPrevMovieInfo(paModeInfo);
      paModeInfo = obj.GetDefaultPrevAxes(paModeInfo);

      if nargin < 2,
        obj.prevAxesModeInfo = paModeInfo;
      end
    end
    
    function ModeInfo = SetPrevMovieInfo(obj,ModeInfo)
      
      if ~obj.hasMovie || ~obj.isPrevAxesModeInfoSet(ModeInfo),
        return;
      end
      
      try
        
        viewi = 1;
        if ModeInfo.iMov == obj.currMovie,
          [im,~,imRoi] = ...
            obj.movieReader(viewi).readframe(ModeInfo.frm,...
            'doBGsub',obj.movieViewBGsubbed,'docrop',~obj.cropIsCropMode);
        else
          mr = MovieReader;
          obj.movieMovieReaderOpen(mr,MovieIndex(ModeInfo.iMov),viewi);
          [im,~,imRoi] = mr.readframe(ModeInfo.frm,...
            'doBGsub',obj.movieViewBGsubbed,'docrop',~obj.cropIsCropMode);
        end
        ModeInfo.im = im;
        ModeInfo.isrotated = false;
        
        % to do: figure out [~,~what to do when there are multiple views
        if ~obj.hasTrx,
          ModeInfo.xdata = imRoi(1:2);
          ModeInfo.ydata = imRoi(3:4);
          %         ModeInfo.xdata = [1,size(ModeInfo.im,2)];
          %         ModeInfo.ydata = [1,size(ModeInfo.im,1)];
        else
          ydir = get(obj.gdata.axes_prev,'YDir');
          if strcmpi(ydir,'normal'),
            pi2sign = -1;
          else
            pi2sign = 1;
          end
          
          [x,y,th] = obj.targetLoc(ModeInfo.iMov,ModeInfo.iTgt,ModeInfo.frm);
          if isnan(th),
            th = -pi/2;
          end
          ModeInfo.A = [1,0,0;0,1,0;-x,-y,1]*[cos(th+pi2sign*pi/2),-sin(th+pi2sign*pi/2),0;sin(th+pi2sign*pi/2),cos(th+pi2sign*pi/2),0;0,0,1];
          ModeInfo.tform = maketform('affine',ModeInfo.A);
          [ModeInfo.im,ModeInfo.xdata,ModeInfo.ydata] = imtransform(ModeInfo.im,ModeInfo.tform,'bicubic');
          ModeInfo.isrotated = true;
        end
        
      catch ME,
        warning(['Error setting reference image information, clearing out reference image.\n',getReport(ME)]);
        obj.clearPrevAxesModeInfo();
      end
      
    end
      
    function [w,h] = GetPrevAxesSizeInPixels(obj)
      units = get(obj.gdata.axes_prev,'Units');
      set(obj.gdata.axes_prev,'Units','pixels');
      pos = get(obj.gdata.axes_prev,'Position');
      set(obj.gdata.axes_prev,'Units',units);
      w = pos(3); h = pos(4);
    end
    
    function ModeInfo = GetDefaultPrevAxes(obj,ModeInfo)
      
      if nargin < 2,
        ModeInfo = obj.prevAxesModeInfo;
      end
      
      borderfrac = .5;
      if ~obj.hasMovie,
        return;
      end
      if ~obj.isPrevAxesModeInfoSet(ModeInfo),
        return;
      end
      if ~isfield(ModeInfo,'isrotated'),
        ModeInfo.isrotated = false;
      end
      lpos = obj.labeledposGTaware;
      viewi = 1;
      ptidx = obj.labeledposIPt2View == viewi;
      poscurr = lpos{ModeInfo.iMov}(ptidx,:,ModeInfo.frm,ModeInfo.iTgt,viewi);
      if obj.hasTrx,
        poscurr = [poscurr,ones(size(poscurr,1),1)]*ModeInfo.A;
        poscurr = poscurr(:,1:2);
      end
            
      minpos = min(poscurr,[],1);
      maxpos = max(poscurr,[],1);
      centerpos = (minpos+maxpos)/2;
      % border defined by borderfrac
      r = max(1,(maxpos-minpos)/2*(1+borderfrac));
      xlim = centerpos(1)+[-1,1]*r(1);
      ylim = centerpos(2)+[-1,1]*r(2);      
      
      [axw,axh] = obj.GetPrevAxesSizeInPixels();
      axszratio = axw/axh;
      dx = diff(xlim);
      dy = diff(ylim);
      limratio = dx / dy;
      % need to extend 
      if axszratio > limratio,
        extendratio = axszratio/limratio;
        xlim = centerpos(1)+[-1,1]*r(1)*extendratio;
      elseif axszratio < limratio,
        extendratio = limratio/axszratio;
        ylim = centerpos(2)+[-1,1]*r(2)*extendratio;
      end
      if isfield(ModeInfo,'dxlim'),
        xlim0 = xlim;
        ylim0 = ylim;
        xlim = xlim + ModeInfo.dxlim;
        ylim = ylim + ModeInfo.dylim;
        % make sure all parts are visible
        if minpos(1) < xlim(1) || minpos(2) < ylim(1) || ...
            maxpos(1) > xlim(2) || maxpos(2) < ylim(2),
          ModeInfo.dxlim = [0,0];
          ModeInfo.dylim = [0,0];
          xlim = xlim0;
          ylim = ylim0;
          fprintf('Templates zoomed axes would not show all labeled points, using default axes.\n');
        end
      else
        ModeInfo.dxlim = [0,0];
        ModeInfo.dylim = [0,0];
      end
      ModeInfo.xlim = xlim;
      ModeInfo.ylim = ylim;
      
      ModeInfo = obj.SetPrevAxesProperties(ModeInfo);
      
%       xdir = get(obj.gdata.axes_curr,'XDir');
%       ydir = get(obj.gdata.axes_curr,'YDir');
%       ModeInfo.axes_curr = struct('XLim',xlim,'YLim',ylim,...
%         'XDir',xdir','YDir',ydir,...
%         'CameraViewAngleMode','auto');
      
      if nargin < 2,
        obj.prevAxesModeInfo = ModeInfo;
      end
      
    end
    
    function ModeInfo = SetPrevAxesProperties(obj,ModeInfo)
      
      
      if nargin < 2,
        ModeInfo = obj.prevAxesModeInfo;
      end
      
      xdir = get(obj.gdata.axes_curr,'XDir');
      ydir = get(obj.gdata.axes_curr,'YDir');
      if ~isfield(ModeInfo,'xlim'),
        xlim = get(obj.gdata.axes_curr,'XLim');
        ylim = get(obj.gdata.axes_curr,'YLim');
      else
        xlim = ModeInfo.xlim;
        ylim = ModeInfo.ylim;
      end
      ModeInfo.axes_curr = struct('XLim',xlim,'YLim',ylim,...
        'XDir',xdir','YDir',ydir,...
        'CameraViewAngleMode','auto');

      if nargin < 2,
        obj.prevAxesModeInfo = ModeInfo;
      end
      
    end
    
    function UpdatePrevAxesLimits(obj)
      
      if obj.prevAxesMode == PrevAxesMode.FROZEN,
        newxlim = get(obj.gdata.axes_prev,'XLim');
        newylim = get(obj.gdata.axes_prev,'YLim');
        dx = newxlim - obj.prevAxesModeInfo.axes_curr.XLim;
        dy = newylim - obj.prevAxesModeInfo.axes_curr.YLim;
        
        obj.prevAxesModeInfo.axes_curr.XLim = newxlim;
        obj.prevAxesModeInfo.axes_curr.YLim = newylim;
        obj.prevAxesModeInfo.dxlim = obj.prevAxesModeInfo.dxlim + dx;
        obj.prevAxesModeInfo.dylim = obj.prevAxesModeInfo.dylim + dy;
      end
      
    end
    
    function UpdatePrevAxesDirections(obj)
      
      xdir = get(obj.gdata.axes_curr,'XDir');
      ydir = get(obj.gdata.axes_curr,'YDir');

      obj.prevAxesModeInfo.axes_curr.XDir = xdir;
      obj.prevAxesModeInfo.axes_curr.YDir = ydir;
      
      set(obj.gdata.axes_prev,'XDir',xdir,'YDir',ydir);
      
      if obj.hasTrx,
        obj.prevAxesModeInfo = obj.SetPrevMovieInfo(obj.prevAxesModeInfo);
        obj.GetDefaultPrevAxes();
        obj.prevAxesFreeze(obj.prevAxesModeInfo);
      end

    end
    
    function InitializePrevAxesTemplate(obj)
      
      islabeled = obj.currFrameIsLabeled();
      if islabeled,
        set(obj.gdata.pushbutton_freezetemplate,'Enable','on');
      else
        set(obj.gdata.pushbutton_freezetemplate,'Enable','off');
      end
      
      if obj.prevAxesMode == PrevAxesMode.FROZEN && ~obj.isPrevAxesModeInfoSet(),
        if islabeled,
          obj.prevAxesFreeze([]);
        end
      end
      
    end
    
    function CheckPrevAxesTemplate(obj)
      
      if obj.prevAxesMode ~= PrevAxesMode.FROZEN || ~obj.isPrevAxesModeInfoSet(),
        return;
      end
      if obj.prevAxesModeInfo.frm == obj.currFrame && obj.prevAxesModeInfo.iMov == obj.currMovie && ...
          obj.prevAxesModeInfo.iTgt == obj.currTarget,
        obj.FixPrevModeInfo();
        obj.setPrevAxesMode(obj.prevAxesMode,obj.prevAxesModeInfo);
      end
      islabeled = obj.currFrameIsLabeled();
      if islabeled,
        set(obj.gdata.pushbutton_freezetemplate,'Enable','on');
      else
        set(obj.gdata.pushbutton_freezetemplate,'Enable','off');
      end
    end
    
    function prevAxesMovieRemap(obj,mIdxOrig2New)
      
      if ~obj.isPrevAxesModeInfoSet(),
        return;
      end
      newIdx = mIdxOrig2New(obj.prevAxesModeInfo.iMov);
      if newIdx == 0,
        obj.clearPrevAxesModeInfo();
        obj.FixPrevModeInfo();
        obj.setPrevAxesMode(obj.prevAxesMode,obj.prevAxesModeInfo);        
      else
        obj.prevAxesModeInfo.iMov = newIdx;
      end
      
    end
    
  end
  methods (Access=private)
    function prevAxesSetLabels(obj,iMov,frm,iTgt,info)
      persistent tfWarningThrownAlready
      
      if nargin < 5,
        isrotated = false;
      else
        isrotated = info.isrotated;
      end
      
      if isempty(frm),
        sz = size(obj.labeledposGTaware{1});
        lpos = nan(sz([1,2]));
        lpostag = false([sz(1),1]);
      else
      lpos = obj.labeledposGTaware;
      lpostag = obj.labeledpostagGTaware;
      lpos = lpos{iMov}(:,:,frm,iTgt);
        if isrotated,
          lpos = [lpos,ones(size(lpos,1),1)]*info.A;
          lpos = lpos(:,1:2);
        end
      lpostag = lpostag{iMov}(:,frm,iTgt);
      end
      ipts = 1:obj.nPhysPoints;
      txtOffset = obj.labelPointsPlotInfo.TextOffset;
      LabelCore.assignLabelCoordsStc(lpos(ipts,:),...
        obj.lblPrev_ptsH(ipts),obj.lblPrev_ptsTxtH(ipts),txtOffset);
      if any(lpostag(ipts))
        if isempty(tfWarningThrownAlready)
          warningNoTrace('Labeler:labelsPrev',...
            'Label tags in previous frame not visualized.');
          tfWarningThrownAlready = true;
        end
      end
    end
  end
  
  %% Labels2/OtherTarget labels
  methods
    
% AL 201806 no callers atm
%     function labels2BulkSet(obj,lpos)
%       assert(numel(lpos)==numel(obj.labeledpos2));
%       for i=1:numel(lpos)
%         assert(isequal(size(lpos{i}),size(obj.labeledpos2{i})))
%         obj.labeledpos2{i} = lpos{i};
%       end
%       if ~obj.gtIsGTMode
%         obj.labels2VizUpdate();
%       end
%     end

    function [tf,lpos2] = labels2IsCurrMovFrmLbled(obj,iTgt)
      % tf: scalar logical, true if tracker has results/predictions for 
      %   currentMov/frm/iTgt 
      % lpos2: [nptsx2] landmark coords
     
      PROPS = obj.gtGetSharedProps();
      iMov = obj.currMovie;
      frm = obj.currFrame;
      lpos2 = obj.(PROPS.LPOS2){iMov}(:,:,frm,iTgt);
      tf = any(~isnan(lpos2(:)));      
    end
    
    function labels2SetCurrMovie(obj,lpos)
      % Works in both reg/GT mode
      PROPS = obj.gtGetSharedProps();
      iMov = obj.currMovie;      
      assert(isequal(size(lpos),size(obj.(PROPS.LPOS){iMov})));
      obj.(PROPS.LPOS2){iMov} = lpos;
    end
    
    function labels2Clear(obj)
      % Operates based on current reg/GT mode
      PROPS = obj.gtGetSharedProps();
      PROPLPOS2 = PROPS.LPOS2;
      for i=1:numel(obj.(PROPLPOS2))
        obj.(PROPLPOS2){i}(:) = nan;
      end
      obj.labels2VizUpdate();
    end
    
    function labels2ImportTrkPromptAuto(obj,iMovs)
      % See labelImportTrkPromptAuto().
      % iMovs: works per current GT mode
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmoviesGTaware;
      end      
      obj.labelImportTrkPromptGenericAuto(iMovs,'labels2ImportTrk');
    end
   
    function labels2ImportTrk(obj,iMovs,trkfiles)
      % Works per current GT mode      
      PROPS = obj.gtGetSharedProps;      
      obj.labelImportTrkGeneric(iMovs,trkfiles,PROPS.LPOS2,[],[]);
      obj.labels2VizUpdate();
      RC.saveprop('lastTrkFileImported',trkfiles{end});
    end
    
    function labels2ImportTrkCurrMov(obj)
      % Try to import default trk file for current movie into labels2. If
      % the file is not there, error.      
      if ~obj.hasMovie
        error('Labeler:nomov','No movie is loaded.');
      end
      obj.labels2ImportTrkPromptAuto(obj.currMovie);
    end
    
    function labels2ExportTrk(obj,iMovs,varargin)
      % Export label2 data to trk files.
      %
      % iMov: optional, indices into (rows of) .movieFilesAllGTaware to 
      %   export. Defaults to 1:obj.nmoviesGTaware.
      
      [trkfiles,rawtrkname] = myparse(varargin,...
        'trkfiles',[],... % [nMov nView] cellstr, fullpaths to trkfilenames to export to
        'rawtrkname',[]... % string, rawname to apply over iMovs to generate trkfiles
        );
      
      if exist('iMovs','var')==0
        iMovs = 1:obj.nmoviesGTaware;
      end
                
      [tfok,trkfiles] = obj.resolveTrkfilesVsTrkRawname(iMovs,trkfiles,...
        rawtrkname,{});
      if ~tfok
        return;
      end

      PROPS = obj.gtGetSharedProps;
      obj.labelExportTrkGeneric(iMovs,trkfiles,PROPS.LPOS2,[],[]);
    end
    
    function colors = Set2PointColors(obj,colors)
      
      colors = colors(obj.labeledposIPt2Set,:);
      
    end
    
    function colors = LabelPointColors(obj,idx)
      
      colors = obj.Set2PointColors(obj.labelPointsPlotInfo.Colors);
      if nargin > 1,
        colors = colors(idx,:);
      end
      
    end
    
    function colors = PredictPointColors(obj)
      colors = obj.Set2PointColors(obj.predPointsPlotInfo.Colors);
    end
    
    function labels2TrkVizInit(obj)
      % Initialize trkViz for .labeledpos2, .trkRes*
      
      tv = obj.labeledpos2trkViz;
      if ~isempty(tv)
        tv.delete();
      end      
      tv = TrackingVisualizerMT(obj,'labeledpos2');
      tv.vizInit();
      obj.labeledpos2trkViz = tv;
    end
    
    function trkResVizInit(obj)
      for i=1:numel(obj.trkResViz)
        tv = obj.trkResViz{i};
        if isempty(tv.lObj)
          tv.postLoadInit(obj);
        else
          tv.vizInit('postload',true);
        end
      end
    end
    
    function labels2VizUpdate(obj,varargin)
      % update trkres from lObj.labeledpos
      
      [dotrkres,setlbls,setprimarytgt] = myparse(varargin,...
        'dotrkres',false,...
        'setlbls',true,...
        'setprimarytgt',false ...
        );
      
      iTgt = obj.currTarget;
      tv = obj.labeledpos2trkViz;

      if setlbls
        iMov = obj.currMovie;
        frm = obj.currFrame;
        lpos2 = reshape(obj.labeledpos2GTaware{iMov}(:,:,frm,:),...
          [obj.nLabelPoints,2,obj.nTargets]);
        % no lpos2 occ!
        lpostag = false(obj.nLabelPoints,obj.nTargets);
        tv.updateTrackRes(lpos2,lpostag);
      end
      if setprimarytgt
        tv.updatePrimary(iTgt);        
      end
      
      if dotrkres
        trkres = obj.trkResGTaware;
        trvs = obj.trkResViz;
        nTR = numel(trvs);
        for iTR=1:nTR
          if ~isempty(trkres{iMov,1,iTR})
            tObjsAll = trkres(iMov,:,iTR);
            
            trkP = cell(obj.nview,1);
            tfHasRes = true;
            for ivw=1:obj.nview
              tObj = tObjsAll{ivw};
              ifrm = find(frm==tObj.pTrkFrm);
              iitgt = find(iTgt==tObj.pTrkiTgt);
              if ~isempty(ifrm) && ~isempty(iitgt)
                trkP{ivw} = tObj.pTrk(:,:,ifrm,iitgt);
              else
                tfHasRes = false;
                break;
              end
            end
            if tfHasRes            
              trkP = cat(1,trkP{:}); % [nlabelpoints x 2 x nfrm x ntgt]
              trv = trvs{iTR};
              trv.updateTrackRes(trkP);
                  % % From DeepTracker.getPredictionCurrFrame
                  % AL20160502: When changing movies, order of updates to
                  % % lObj.currMovie and lObj.currFrame is unspecified. currMovie can
                  % % be updated first, resulting in an OOB currFrame; protect against
                  % % this.
                  % frm = min(frm,size(xyPCM,3));
                  % xy = squeeze(xyPCM(:,:,frm,:)); % [npt x d x ntgt]
            end
          end
        end
      end
    end
    
    function labels2VizShowHideUpdate(obj)
      tfHide = obj.labels2Hide;
      txtprops = obj.impPointsPlotInfo.TextProps;
      tfHideTxt = strcmp(txtprops.Visible,'off');      
      tv = obj.labeledpos2trkViz;
      tv.setAllShowHide(tfHide,tfHideTxt,obj.labels2ShowCurrTargetOnly);
    end
    
    function labels2VizShow(obj)
      obj.labels2Hide = false;
      obj.labels2VizShowHideUpdate();
    end
    
    function labels2VizHide(obj)
      obj.labels2Hide = true;
      obj.labels2VizShowHideUpdate();
    end
    
    function labels2VizToggle(obj)
      if obj.labels2Hide
        obj.labels2VizShow();
      else
        obj.labels2VizHide();
      end
    end

    function labels2VizSetShowCurrTargetOnly(obj,tf)
      obj.labels2ShowCurrTargetOnly = tf;
      obj.labels2VizShowHideUpdate();
    end
     
  end
  
  methods % OtherTarget
    
    function labelsOtherTargetShowIdxs(obj,iTgts)
      
      % AL 20190605 maybe remove
      return
      
      frm = obj.currFrame;
      lpos = obj.labeledposCurrMovie;
      lpos = squeeze(lpos(:,:,frm,iTgts)); % [npts x 2 x numel(iTgts)]

      npts = obj.nLabelPoints;     
      hPts = obj.lblOtherTgts_ptsH;
      for ipt=1:npts
        xnew = squeeze(lpos(ipt,1,:));
        ynew = squeeze(lpos(ipt,2,:));
        set(hPts(ipt),'XData',[hPts(ipt).XData xnew'],...
                      'YData',[hPts(ipt).YData ynew']);
      end
    end
    
    function labelsOtherTargetHideAll(obj)
      
      % AL 20190605 maybe remove
      return

      npts = obj.nLabelPoints;
      hPts = obj.lblOtherTgts_ptsH;
      for ipt=1:npts
        set(hPts(ipt),'XData',[],'YData',[]);
      end      
    end
    
  end
  
  %% Emp PDF
  methods
    function updateFGEmpiricalPDF(obj,varargin)
      
      if ~obj.hasTrx
        error('Method only supported for projects with trx.');
      end
      if obj.gtIsGTMode
        error('Method is not supported in GT mode.');
      end
      if obj.cropProjHasCrops
        % in general projs with trx would never use crop info 
        error('Method unsupported for projects with cropping info.');
      end
      if obj.nview>1
        error('Method is not supported for multiple views.');
      end
      tObj = obj.tracker;
      if isempty(tObj)
        error('Method only supported for projects with trackers.');
      end
      
      prmPP = obj.preProcParams;
      prmBackSub = prmPP.BackSub;
      prmNborMask = prmPP.NeighborMask;
      if isempty(prmBackSub.BGType) || isempty(prmBackSub.BGReadFcn)
        error('Computing the empirical foreground PDF requires a background type and background read function to be defined in the tracking parameters.');
      end
      if ~prmNborMask.Use
        warningNoTrace('Neighbor masking is currently not turned on in your tracking parameters.');
      end
      
      % Start with all labeled rows. Prefer these b/c user apparently cares
      % more about these frames
      wbObj = WaitBarWithCancel('Empirical Foreground PDF');
      oc = onCleanup(@()delete(wbObj));
      tblMFTlbled = obj.labelGetMFTableLabeled('wbObj',wbObj);
      if wbObj.isCancel
        return;
      end
      roiRadius = prmPP.TargetCrop.Radius;
      tblMFTlbled = obj.labelMFTableAddROITrx(tblMFTlbled,roiRadius);

      amu = mean(tblMFTlbled.aTrx);
      bmu = mean(tblMFTlbled.bTrx);
      %fprintf('amu/bmu: %.4f/%.4f\n',amu,bmu);
      
      % get stuff we will need for movies: movieReaders, bgimages, etc
      movieStuff = cell(obj.nmovies,1);
      iMovsLbled = unique(tblMFTlbled.mov);
      for iMov=iMovsLbled(:)'        
        s = struct();
        
        mr = MovieReader();
        obj.movieMovieReaderOpen(mr,MovieIndex(iMov),1);
        s.movRdr = mr;
        
        trxfname = obj.trxFilesAllFull{iMov,1};
        movIfo = obj.movieInfoAll{iMov};
        [s.trx,s.frm2trx] = obj.getTrx(trxfname,movIfo.nframes);
                
        movieStuff{iMov} = s;
      end    
      
      hFigViz = figure; %#ok<NASGU>
      ax = axes;
    
      xroictr = -roiRadius:roiRadius;
      yroictr = -roiRadius:roiRadius;
      [xgrid,ygrid] = meshgrid(xroictr,yroictr); % xgrid, ygrid give coords for each pixel where (0,0) is the central pixel (at target)
      pdfRoiAcc = zeros(2*roiRadius+1,2*roiRadius+1);
      nAcc = 0;
      n = height(tblMFTlbled);
      for i=1:n
        trow = tblMFTlbled(i,:);
        iMov = trow.mov;
        frm = trow.frm;
        iTgt = trow.iTgt;
        
        sMovStuff = movieStuff{iMov};
        
        [tflive,trxxs,trxys,trxths,trxas,trxbs] = ...
          PxAssign.getTrxStuffAtFrm(sMovStuff.trx,frm);
        assert(isequal(tflive(:),sMovStuff.frm2trx(frm,:)'));

        % Skip roi if it contains more than 1 trxcenter.
        roi = trow.roi; % [xlo xhi ylo yhi]
        roixlo = roi(1);
        roixhi = roi(2);
        roiylo = roi(3);
        roiyhi = roi(4);
        tfCtrInRoi = roixlo<=trxxs & trxxs<=roixhi & roiylo<=trxys & trxys<=roiyhi;
        if nnz(tfCtrInRoi)>1
          continue;
        end
      
        % In addition run CC pxAssign and keep only the central CC to get
        % rid of any objects at the periphery
        imdiff = sMovStuff.movRdr.readframe(frm,'doBGsub',true);
        assert(isa(imdiff,'double'));
        imbwl = PxAssign.asgnCCcore(imdiff,sMovStuff.trx,frm,prmNborMask.FGThresh);
        xTgtCtrRound = round(trxxs(iTgt));
        yTgtCtrRound = round(trxys(iTgt));
        ccKeep = imbwl(yTgtCtrRound,xTgtCtrRound);
        if ccKeep==0
          warningNoTrace('Unexpected non-foreground pixel for (mov,frm,tgt)=(%d,%d,%d) at (r,c)=(%d,%d).',...
            iMov,frm,iTgt,yTgtCtrRound,xTgtCtrRound);
        else
          imfgUse = zeros(size(imbwl));
          imfgUse(imbwl==ccKeep) = 1;                    
          imfgUseRoi = padgrab(imfgUse,0,roiylo,roiyhi,roixlo,roixhi);
          
          th = trxths(iTgt);
          a = trxas(iTgt);
          b = trxbs(iTgt);
          imforebwcanon = readpdf(imfgUseRoi,xgrid,ygrid,xgrid,ygrid,0,0,-th);
          xfac = a/amu;
          yfac = b/bmu;
          imforebwcanonscale = interp2(xgrid,ygrid,imforebwcanon,...
            xgrid*xfac,ygrid*yfac,'linear',0);
          
          imshow(imforebwcanonscale,'Parent',ax);
          tstr = sprintf('row %d/%d',i,n);
          title(tstr,'fontweight','bold','interpreter','none');
          drawnow;
          
          pdfRoi = imforebwcanonscale/sum(imforebwcanonscale(:)); % each row equally weighted
          pdfRoiAcc = pdfRoiAcc + pdfRoi;
          nAcc = nAcc + 1;
        end
      end
      
      fgpdf = pdfRoiAcc/nAcc;
      
      imshow(fgpdf,[],'xdata',xroictr,'ydata',yroictr);
      colorbar;
      tstr = sprintf('N=%d, amu=%.3f, bmu=%.3f. FGThresh=%.2f',...
        nAcc,amu,bmu,prmNborMask.FGThresh);
      title(tstr,'fontweight','bold','interpreter','none');
      
      obj.fgEmpiricalPDF = struct(...
        'amu',amu,'bmu',bmu,...
        'xpdfctr',xroictr,'ypdfctr',yroictr,...        
        'fgpdf',fgpdf,...
        'n',nAcc,...
        'roiRadius',roiRadius,...
        'prmBackSub',prmBackSub,...
        'prmNborMask',prmNborMask);
    end
  end
  
  
  %% Util
  methods
    
    function tblConcrete = mftTableConcretizeMov(obj,tbl)
      % tbl: MFTable where .mov is MovieIndex array
      %
      % tblConcrete: Same table where .mov is [NxNview] cellstr; column 
      %   .trxFile is also added when appropriate
      
      assert(isa(tbl,'table'));

      n = height(tbl);
      tblConcrete = tbl;
      mIdx = tblConcrete.mov;
      assert(isa(mIdx,'MovieIndex'));
      [iMovAbs,tfGTRow] = mIdx.get();
      tfRegRow = ~tfGTRow;
      assert(~any(iMovAbs==0));
      movStr = cell(n,obj.nview);
      movStr(tfRegRow,:) = obj.movieFilesAllFull(iMovAbs(tfRegRow),:); % [NxnView] 
      movStr(tfGTRow,:) = obj.movieFilesAllGTFull(iMovAbs(tfGTRow),:);
      tblConcrete.mov = movStr;

      if obj.hasTrx && obj.nview==1
        trxFile = repmat({''},n,1);
        trxFile(tfRegRow) = obj.trxFilesAllFull(iMovAbs(tfRegRow),:);
        trxFile(tfGTRow) = obj.trxFilesAllGTFull(iMovAbs(tfGTRow),:);
        tblConcrete = [tblConcrete table(trxFile)];
      end
    end
    
    function genericInitLabelPointViz(obj,hProp,hTxtProp,ax,plotIfo)
      deleteValidHandles(obj.(hProp));
      obj.(hProp) = gobjects(obj.nLabelPoints,1);
      if ~isempty(hTxtProp)
        deleteValidHandles(obj.(hTxtProp));
        obj.(hTxtProp) = gobjects(obj.nLabelPoints,1);
      end
      
      markerPVcell = struct2pvs(plotIfo.MarkerProps);
      textPVcell = struct2pvs(plotIfo.TextProps);
      
      % any extra plotting parameters
      allowedPlotParams = {'HitTest' 'PickableParts'};
      ism = ismember(cellfun(@lower,allowedPlotParams,'Uni',0),...
                     cellfun(@lower,fieldnames(plotIfo),'Uni',0));
      extraParams = {};
      for i = find(ism)
        extraParams = [extraParams,{allowedPlotParams{i},plotIfo.(allowedPlotParams{i})}]; %#ok<AGROW>
      end

      for i = 1:obj.nLabelPoints
        obj.(hProp)(i) = plot(ax,nan,nan,markerPVcell{:},...
          'Color',plotIfo.Colors(i,:),...
          'UserData',i,...
          extraParams{:},...
          'Tag',sprintf('Labeler_%s_%d',hProp,i));
        if ~isempty(hTxtProp)
          obj.(hTxtProp)(i) = text(nan,nan,num2str(i),'Parent',ax,...
            textPVcell{:},'Color',plotIfo.Colors(i,:),...
            'PickableParts','none',...
            'Tag',sprintf('Labeler_%s_%d',hTxtProp,i));
        end
      end      
    end
   
    function SetStatus(obj,s,varargin)
      
      try
        if isfield(obj.gdata,'SetStatusFun'),
          obj.gdata.SetStatusFun(obj.gdata,s,varargin{:});
        else
          fprintf(['Status: ',s,'...\n']);
        end
      catch
        fprintf(['Status: ',s,'...\n']);
      end
      
    end

    function ClearStatus(obj,varargin)
      
      try
        if isfield(obj.gdata,'ClearStatusFun'),
          obj.gdata.ClearStatusFun(obj.gdata,varargin{:});
        else
          fprintf('Done.\n');
        end
      catch
        fprintf('Done.\n');
      end
    end
    
    function setStatusBarTextWhenClear(obj,s,varargin)
      
      if isfield(obj.gdata,'SetStatusBarTextWhenClearFun'),
        obj.gdata.SetStatusBarTextWhenClearFun(obj.gdata,s,varargin{:});
      else
        fprintf(['Ready status: ',s,'...\n']);
      end
      
    end
    
    function raiseAllFigs(obj)
      h = obj.gdata.figs_all;
      arrayfun(@figure,h);
    end
    
    function addDepHandle(obj,h)
      handles = obj.gdata;
      handles.depHandles(end+1,1) = h;
      guidata(obj.hFig,handles);      
    end
    
    function v = allMovIdx(obj)
      
      v = MovieIndex(1:obj.nmoviesGTaware,obj.gtIsGTMode);
      
    end
    
    % make a toTrack struct from selected movies in the project amenable to
    % TrackBatchGUI
    function toTrack = mIdx2TrackList(obj,mIdx)
      
      if nargin < 2 || isempty(mIdx),
        mIdx = obj.allMovIdx();
      end
      nget = numel(mIdx);
      toTrack = struct(...
        'movfiles', {cell(nget,obj.nview)},...
        'trkfiles', {cell(nget,obj.nview)},...
        'trxfiles', {cell(nget,obj.nview)},...
        'cropRois', {cell(nget,obj.nview)},...
        'calibrationfiles', {cell(nget,1)},... %        'calibrationdata',{cell(nget,1)},...
        'targets', {cell(nget,1)},...
        'f0s', {cell(nget,1)},...
        'f1s', {cell(nget,1)});
      toTrack.movfiles = obj.getMovieFilesAllFullMovIdx(mIdx);
      toTrack.trxfiles = obj.getTrxFilesAllFullMovIdx(mIdx);
      for i = 1:nget,        
        if obj.cropProjHasCrops,
          [tfhascrop,roi] = obj.cropGetCropMovieIdx(mIdx(i));
          if tfhascrop,
            for j = 1:obj.nview,
              toTrack.cropRois{i,j} = roi(j,:);
            end
          end
        end
        vcd = obj.getViewCalibrationDataMovIdx(mIdx(i));
        if ~isempty(vcd),
          toTrack.calibrationfiles{i} = vcd.sourceFile;
          %toTrack.calibrationdata{i} = vcd;
        end
      end
      
      rawname = obj.defaultExportTrkRawname();
      [tfok,trkfiles] = obj.getTrkFileNamesForExportUI(toTrack.movfiles,rawname,'noUI',true);
      if tfok,
        toTrack.trkfiles = trkfiles;
      end

    end
    
  end
  
  methods (Static)
    
     function [iMov1,iMov2] = identifyCommonMovSets(movset1,movset2)
      % Find common rows (moviesets) in two sets of moviesets
      %
      % movset1: [n1 x nview] cellstr of fullpaths. Call 
      %   FSPath.standardPath and FSPath.platformizePath on these first
      % movset2: [n2 x nview] "
      %
      % IMPORTANT: movset1 is allowed to have duplicate rows/moviesets,
      % but dupkicate rows/movsets in movset2 will be "lost".
      % 
      % iMov1: [ncommon x 1] index vector into movset1. Could be empty
      % iMov2: [ncommon x 1] " movset2 "
      %
      % If ncommon>0, then movset1(iMov1,:) is equal to movset2(iMov2,:)
      
      assert(size(movset1,2)==size(movset2,2));
      nvw = size(movset1,2);
      
      for ivw=nvw:-1:1
        [tf(:,ivw),loc(:,ivw)] = ismember(movset1(:,ivw),movset2(:,ivw));
      end
      
      tf = all(tf,2);
      iMov1 = find(tf);      
      nCommon = numel(iMov1);
      iMov2 = zeros(nCommon,1);
      for i=1:nCommon
        locUn = unique(loc(iMov1(i),:));
        if isscalar(locUn)
          iMov2(i) = locUn;
        else
          % warningNoTrace('Inconsistency in movielists detected across views.');
          % iMov2(i) initted to 0
        end
      end
      
      tfRm = iMov2==0;
      iMov1(tfRm,:) = [];
      iMov2(tfRm,:) = [];
     end
    
  end

end
