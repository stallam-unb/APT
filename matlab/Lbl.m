classdef Lbl
  methods (Static) % ma train package
    
    function vizLoc(loc,packdir,varargin)
      % Visualize 'loc' data structure (one row per labeled mov,frm,tgt) 
      % from training package.
      %
      % loc: loc structure as output by Lbl.loadPack
      % packdir: package dir (contains images)

      [scargs,ttlargs] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'} ...
        );
      
      hfig = figure(11);
      
      idmfun = unique({loc.idmovfrm}');
      nmfun = numel(idmfun);
      for iidmf=1:nmfun
        idmf = idmfun{iidmf};
        is = find(strcmp({loc.idmovfrm}',idmf));
        
        imf = fullfile(packdir,'im',[idmf '.png']);
        im = imread(imf);
        
        clf;
        ax = axes;
        imagesc(im);
        colormap gray;
        hold on;
        axis square;
        
        for j = is(:)'
          s = loc(j);
          xy = reshape(s.p,[],2);        
          scatter(xy(:,1),xy(:,2),scargs{:});        
          plot(s.roi([1:4 1]),s.roi([5:8 5]),'r-','linewidth',2);
        end
        
        tstr = sprintf('%s: %d tgts',idmf,numel(is));
        title(tstr,ttlargs{:});
        input(idmf);
      end        
    end
    
    function vizLocg(locg,packdir,varargin)
      % Visualize 'locg' data structure (one row per labeled mov,frm) 
      % from training package.
      %
      % locg: loc structure as output by Lbl.loadPack
      % packdir: package dir (contains images)

      [scargs,ttlargs] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'} ...
        );
      
      hfig = figure(11);
      
      nfrm = numel(locg.locdata);
      for ifrm=1:nfrm
        s = locg.locdata(ifrm);
        imf = fullfile(packdir,s.img);
        im = imread(imf);
        
        clf;
        ax = axes;
        imagesc(im);
        colormap gray;
        hold on;
        axis square;
        
        for itgt=1:s.ntgt
          xy = reshape(s.pabs(:,itgt),[],2);
          scatter(xy(:,1),xy(:,2),scargs{:});
          plot(s.roi([1:4 1],itgt),s.roi([5:8 5],itgt),'r-','linewidth',2);
        end
        
        tstr = sprintf('%s: %d tgts',s.id,s.ntgt);
        title(tstr,ttlargs{:});
        input(tstr);
      end        
    end
    
    function s = hlpLoadJson(jsonfile)
      jse = readtxtfile(jsonfile);
      s = jsondecode(jse{1});
      fprintf(1,'loaded %s\n',jsonfile);
    end
    function [slbl,tp,loc,locg] = loadPack(packdir)
      % Load training package into MATLAB data structures
      %
      % slbl: 'stripped lbl' struct
      % tp: one-row-per-movie struct. Maybe a useful format for metadata 
      %   or bookkeeping purposes.
      % loc: one-row-per-labeled-(mov,frm,tgt) struct. Intended to be
      %   primary MA keypt data structure.
      % loccc: one-row-per-labeled-cluster. Experimental, may not be
      %   useful for DL/py backend.
      %
      % Note tp, loc, loccc contain equivalent info just in different
      % formats.
       
      dd = dir(fullfile(packdir,'*.lbl'));
      assert(isscalar(dd));
      lblsf = fullfile(packdir,dd.name);
      slbl = load(lblsf,'-mat');
      fprintf(1,'loaded %s\n',lblsf);
      
      tpf = fullfile(packdir,'trnpack.json');
      tp = Lbl.hlpLoadJson(tpf);

      locf = fullfile(packdir,'loc0.json');
      loc = Lbl.hlpLoadJson(locf);

      locf = fullfile(packdir,'loc.json');
      locg = Lbl.hlpLoadJson(locf);

%       locf = fullfile(packdir,'locclus.json');
%       locjse = readtxtfile(locf);
%       loccc = jsondecode(locjse{1});
%       fprintf(1,'loaded %s\n',locf);
    end
    
    function hlpSaveJson(s,packdir,jsonoutf)
      j = jsonencode(s);
      jsonoutf = fullfile(packdir,jsonoutf);
      fh = fopen(jsonoutf,'w');
      fprintf(fh,'%s\n',j);
      fclose(fh);
      fprintf(1,'Wrote %s.\n',jsonoutf);
    end
    function [slbl,tp,loc,locg] = genWriteTrnPack(lObj,packdir)
      % Generate training package. Write contents (raw images and keypt 
      % jsons) to packdir.
      
      if exist(packdir,'dir')==0
        mkdir(packdir);
      end
      
      tObj = lObj.tracker;
      tObj.setAllParams(lObj.trackGetParams()); % does not set skel, flipLMEdges
      slbl = tObj.trnCreateStrippedLbl();
      slbl = Lbl.compressStrippedLbl(slbl,'ma',true);
      
      fsinfo = lObj.projFSInfo;
      [lblP,lblS] = myfileparts(fsinfo.filename);
      sfname = sprintf('%s_%s.lbl',lblS,tObj.algorithmName);
      sfname = fullfile(packdir,sfname);
      save(sfname,'-mat','-struct','slbl');
      fprintf(1,'Saved %s\n',sfname);

      tp = Lbl.aggregateLabelsAddRoi(lObj);
      [loc,locg,loccc] = Lbl.genLocs(tp,lObj.movieInfoAll);
      Lbl.writeims(loc,packdir);
        
      % trnpack: one row per mov
      jsonoutf = 'trnpack.json';
      Lbl.hlpSaveJson(tp,packdir,jsonoutf);
      
      % loc: one row per labeled tgt
      jsonoutf = 'loc0.json';
      Lbl.hlpSaveJson(loc,packdir,jsonoutf);

      % loc: one row per frm
      jsonoutf = 'loc.json';
      s = struct();
      s.movies = lObj.movieFilesAllFull;
      s.splitnames = {'trn'};
      s.locdata = locg;
      Lbl.hlpSaveJson(s,packdir,jsonoutf);

%       % loccc: one row per cluster
%       jsonoutf = 'locclus.json';
%       Lbl.hlpSaveJson(loccc,packdir,jsonoutf);      
    end
    
    function sagg = aggregateLabelsAddRoi(lObj)
      nmov = numel(lObj.labels);
      sagg = cell(nmov,1);
      for imov=1:nmov
        s = lObj.labels{imov};
        s.mov = lObj.movieFilesAllFull{imov};
        
        %% gen rois, bw
        n = size(s.p,2);
        s.roi = nan(8,n);
        fprintf(1,'mov %d: %d labeled frms.\n',imov,n);
        for i=1:n
          p = s.p(:,i);
          xy = Shape.vec2xy(p);
          roi = lObj.maGetRoi(xy);
          s.roi(:,i) = roi(:);
        end
        
        sagg{imov} = s;
      end
      sagg = cell2mat(sagg);
    end
    function [sloc,slocg,sloccc] = genLocs(sagg,movInfoAll)
      assert(numel(sagg)==numel(movInfoAll));
      nmov = numel(sagg);
      sloc = [];
      slocg = [];
      sloccc = [];
      for imov=1:nmov
        s = sagg(imov);
        movifo = movInfoAll{imov};
        imsz = [movifo.info.nr movifo.info.nc];
        fprintf(1,'mov %d (sz=%s): %s\n',imov,mat2str(imsz),s.mov);
        
        slocI = Lbl.genLocsI(s,imov);
        slocgI = Lbl.genLocsGroupedI(s,imov);
        slocccI = Lbl.genCropClusteredLocsI(s,imsz,imov);
        
        sloc = [sloc; slocI]; %#ok<AGROW>
        slocg = [slocg; slocgI]; %#ok<AGROW>
        sloccc = [sloccc; slocccI]; %#ok<AGROW>
      end
    end
    function [sloc] = genLocsI(s,imov)
      sloc = [];
      nrows = size(s.p,2);
      for j=1:nrows        
        f = s.frm(j);
        itgt = s.tgt(j);
        ts = s.ts(:,j);
        occ = s.occ(:,j);
        roi = s.roi(:,j);
        sloctmp = struct(...
          'id',sprintf('mov%04d_frm%08d_tgt%03d',imov,f,itgt),...
          'idmovfrm',sprintf('mov%04d_frm%08d',imov,f),...
          'imov',imov,...
          'mov',s.mov,...
          'frm',f,...
          'itgt',itgt,...
          'roi',roi,...
          'p',s.p(:,j), ...
          'occ',occ, ...
          'ts',ts ...
          );
        sloc = [sloc; sloctmp]; %#ok<AGROW>
      end
    end
    function [slocgrp] = genLocsGroupedI(s,imov,varargin)
      % s: scalar element of 'sagg', ie labels data structure for one movie.
      % imov: movie index, only used for metadata
      
      imgpat = myparse(varargin,...
        'imgpat','im/%s.png' ...
        );
      
      s = Labels.addsplitsifnec(s);

      slocgrp = [];
      frmsun = unique(s.frm);
      nfrmsun = numel(frmsun);
      for ifrmun=1:nfrmsun
        f = frmsun(ifrmun);
        j = find(s.frm==f);
        ntgt = numel(j);
                    
        % Dont include numtgts, eg what if a target is added to an
        % existing frame.
        basefS = sprintf('mov%04d_frm%08d',imov,f);
        img = sprintf(imgpat,basefS);
        sloctmp = struct(...
          'id',basefS,...
          'img',img,...
          'imov',imov,... % 'mov',s.mov,...
          'frm',f,...
          'ntgt',ntgt,...
          'split',s.split(j),...
          'itgt',s.tgt(j),...
          'roi',s.roi(:,j),...
          'pabs',s.p(:,j), ...
          'occ',s.occ(:,j), ...
          'ts',s.ts(:,j) ...
          );
        slocgrp = [slocgrp; sloctmp]; %#ok<AGROW>
      end
    end
    function [sloccc] = genCropClusteredLocsI(s,imsz,imov)
      % s: scalar element of 'sagg', ie labels data structure for one movie.
      % imsz: [nr nc]
      % imov: movie index, only used for metadata
      
      sloccc = [];
      frmsun = unique(s.frm);
      nfrmsun = numel(frmsun);
      for ifrmun=1:nfrmsun
        f = frmsun(ifrmun);
        idx = find(s.frm==f);
        ntgt = numel(idx);
        %itgt = s.tgt(idx);
        mask = zeros(imsz);
        for j=idx(:)'
          bw = poly2mask(s.roi(1:4,j),s.roi(5:8,j),imsz(1),imsz(2));
          mask(bw) = j; % this may 'overwrite' prev nonzero vals but this
          % is ok. In very rare cases, multiple targets may completely
          % obscure/cover a previous target/roi but this should be
          % exceedingly rare.
        end
        
        % mask is now a label matrix where the labels are the j vals or
        % indices into s.
        cc = bwconncomp(mask);
        ncc = cc.NumObjects;
        % set of tgts/js in each cc
        js = cellfun(@(x)unique(mask(x)),cc.PixelIdxList,'uni',0);
        jsall = cat(1,js{:});
        % Each tgt/j should appear in precisely one cc
        assert(numel(jsall)==ntgt && isequal(sort(jsall),sort(idx)));
        
        for icc=1:ncc
          jcc = js{icc};
          ntgtcc = numel(jcc);
          itgtcc = s.tgt(jcc);
          xyf = reshape(s.p(:,jcc),s.npts,2,ntgtcc); % shapes for all tgts in this cc
          ts = reshape(s.ts(:,jcc),s.npts,ntgtcc); % ts "
          occ = reshape(s.occ(:,jcc),s.npts,ntgtcc); % estocc "
          
          [rcc,ccc] = ind2sub(size(mask),cc.PixelIdxList{icc});
          c0 = min(ccc);
          c1 = max(ccc);
          r0 = min(rcc);
          r1 = max(rcc);
          
          roicrop = [c0 c1 r0 r1];
          xyfcrop = xyf;
          xyfcrop(:,1,:) = xyfcrop(:,1,:)-c0+1;
          xyfcrop(:,2,:) = xyfcrop(:,2,:)-r0+1;                    
          
          % Dont include numtgts, eg what if a target is added to an
          % existing frame.
          basefS = sprintf('mov%04d_frm%08d_cc%03d',imov,f,icc);
          %basefSimfrm = sprintf('mov%04d_frm%08d',imov,f);
          
          % one row per CC
          sloctmp = struct(...
            'id',basefS,...
            'imov',imov,...
            'mov',s.mov,...
            'frm',f,...
            'cc',icc,...
            'ntgt',ntgtcc,...
            'itgt',itgtcc,...
            'roicrop',roicrop, ...
            'xyabs',xyf, ...
            'xycrop',xyfcrop, ...
            'occ',occ, ...
            'ts',ts ...
            );
          sloccc = [sloccc; sloctmp]; %#ok<AGROW>
        end
      end
    end
    
    function writeims(sloc,packdir)
      
      SUBDIRIM = 'im';
      sdir = SUBDIRIM;
      if exist(fullfile(packdir,sdir),'dir')==0
        mkdir(packdir,sdir);
      end
      
      mr = MovieReader;
      for i=1:numel(sloc)
        s = sloc(i);
        
        % Expect sloc to be in 'movie order'
        if ~strcmp(s.mov,mr.filename)
          mr.close();
          mr.open(s.mov);
          fprintf(1,'Opened movie: %s\n',s.mov);
        end
      
        imfrmf = fullfile(packdir,sdir,[s.idmovfrm '.png']);
        if exist(imfrmf,'file')>0
          fprintf(1,'Skipping, image already exists: %s\n',imfrmf);
        else
          imfrm = mr.readframe(s.frm);
          imwrite(imfrm,imfrmf);
          fprintf(1,'Wrote %s\n',imfrmf);
        end
%         sloc(i).imfile = imfrmf;
      end
    end
%     function writeimscc(sloccc,packdir)
%       
%       SUBDIRIM = 'imcc';
%       sdir = SUBDIRIM;
%       if exist(fullfile(packdir,sdir),'dir')==0
%         mkdir(packdir,sdir);
%       end
%       
%       
%       mr = MovieReader;
%       for i=1:numel(sloccc)
%         s = sloccc(i);
%         
%         % Expect sloc to be in 'movie order'
%         if ~strcmp(s.mov,mr.filename)
%           mr.close();
%           mr.open(s.mov);
%           fprintf(1,'Opened movie: %s\n',s.mov);
%         end
%       
%         imfrm = mr.readframe(f);
%         imfrmmask = imfrm;
%         imfrmmask(~maskcc) = 0;
%         imfrmmaskcrop = imfrmmask(r0:r1,c0:c1);
%         if writeims
%           basefSpng = [basefS '.png'];
%           basefSimfrmpng = [basefSimfrm '.png'];
%           %maskf = fullfile(packdir,SUBDIRMASK,basefSpng);
%           imfrmf = fullfile(packdir,SUBDIRIM,basefSimfrmpng);
%           %imfrmmaskf = fullfile(packdir,SUBDIRIMMASK,basefSpng);
%           imfrmmaskcropf = fullfile(packdir,SUBDIRIMMASKC,basefSpng);
%           
%           %imwrite(mask,maskf);
%           if icc==1
%             imwrite(imfrm,imfrmf);
%           end
%           %imwrite(imfrmmask,imfrmmaskf);
%           imwrite(imfrmmaskcrop,imfrmmaskcropf);
%           fprintf(1,'Wrote files for %s...\n',basefS);s
%         else
%           fprintf(1,'Didnt write files for %s...\n',basefS);
%         end
%       end
%     end
  end
  methods (Static) % stripped lbl
    function s = createStrippedLblsUseTopLevelTrackParams(lObj,iTrkers,...
        varargin)
      % Create/save a series of stripped lbls based on current Labeler proj
      %
      % lObj: Labeler obj with proj loaded
      % iTrkers: vector of tracker indices for which stripped lbl will be 
      %   saved
      %
      % s: cell array of stripped lbls
      %
      % This method exists bc:
      % - Strictly speaking, stripped lbls are net-specific, as setting
      % base tracking parameters onto a DeepTracker obj has hooks/codepath
      % for mutating params.
      % - Sometimes, you want to generate a stripped lbl from the top-level
      % params which are not yet set on a particular tracker.
      %
      % This method is here rather than Labeler bc Labeler is getting big.
            
      [docompress,dosave] = myparse(varargin,...
        'docompress',true, ...
        'dosave',true ... save stripped lbls (loc printed)
        );
      
      ndt = numel(iTrkers);
      s = cell(ndt,1);
      for idt=1:ndt
        itrker = iTrkers(idt);
        lObj.trackSetCurrentTracker(itrker);
        tObj = lObj.tracker;
  
        tObj.setAllParams(lObj.trackGetParams()); % does not set skel, flipLMEdges
        sthis = tObj.trnCreateStrippedLbl();
        if docompress
          sthis = Lbl.compressStrippedLbl(sthis);
        end
        
        s{idt} = sthis;
        
        if dosave
          fsinfo = lObj.projFSInfo;
          [lblP,lblS] = myfileparts(fsinfo.filename);
          sfname = sprintf('%s_%s.lbl',lblS,tObj.algorithmName);
          sfname = fullfile(lblP,sfname);
          save(sfname,'-mat','-struct','sthis');
          fprintf(1,'Saved %s\n',sfname);
        end
      end
      
    end
    function s = compressStrippedLbl(s,varargin)
      ma = myparse(varargin,...
        'ma',false ...
        );
      
      CFG_GLOBS = {'Num'};
      FLDS = {'cfg' 'projname' 'projMacros' 'movieInfoAll' 'cropProjHasCrops' ...
        'trackerClass' 'trackerData'};
      TRACKERDATA_FLDS = {'sPrmAll' 'trnNetTypeString'};
      if ma
        GLOBS = {};
        FLDSRM = {'projMacros'};
      else
        GLOBS = {'labeledpos' 'movieFilesAll' 'trxFilesAll' 'preProcData'};
        FLDSRM = { ... % 'movieFilesAllCropInfo' 'movieFilesAllGTCropInfo' ...
                  'movieFilesAllHistEqLUT' 'movieFilesAllGTHistEqLUT'};
      end
      
      fldscfg = fieldnames(s.cfg);      
      fldscfgkeep = fldscfg(startsWith(fldscfg,CFG_GLOBS));
      s.cfg = structrestrictflds(s.cfg,fldscfgkeep);
      
      s.trackerData{2} = structrestrictflds(s.trackerData{2},TRACKERDATA_FLDS);
      
      flds = fieldnames(s);
      fldskeep = flds(startsWith(flds,GLOBS));
      fldskeep = [fldskeep(:); FLDS(:)];
      fldskeep = setdiff(fldskeep, FLDSRM);
      s = structrestrictflds(s,fldskeep);
    end
  end
end