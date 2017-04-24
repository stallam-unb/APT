classdef OrthoCamCalPair < CalRig
  % A pair of calibrated Orthocams, with calibration info
  
  properties
    tblInt % table of intrinsics
    
    % extrinsics
    r2vec1 % [3x1] 
    t2vec1 % [2x1]
    r2vec2 % [3x1]
    t2vec2 % [2x1]
    rvecs % [nPatx3] IMPORTANT: Note this has a row for EVERY PATTERN not patterns 2..nPat.
    tvecs % [nPatx3]
    
    % extrinsics2 -- these are derived from extrinsics, but for convenience
    % we compute them initially and store them b/c after a transformation
    % it may be figure out eg the sign of n1/n2, there may be numerical
    % issues due to the cameras being at precisely 90deg, etc.
    optCtr1 % [3x1] World coords where cam1 optical axis intersected original WorldSys (cal pattern) at z=0 
    optCtr2 % [3x1] " cam2
    ijkCamWorld1 % [3x3] columns are "CamWorldCoords" i/j/k unit vecs in WorldSys for cam1; k is negatice optical axis
    ijkCamWorld2 % "

    calNumPatterns % number of calibration patterns used
    calNumPoints % number of points in calibration pattern
    calWorldPoints % [3xnPts] (x,y,z) of calibration points in PatternWorldSys
    calImPoints % [2xnPtsxnPatx2] (x,y) for each point, pattern, camera
    calPatternFPNs % [nPatx2] cellstr of calibration images
    calTS % timestamp of most recent calibration
  end
  
  properties % CalRig
    nviews = 2;
    viewNames = {'side' 'front'};
    viewSizes = [1024 1024;1024 1024];;    
  end

  methods
    
    function obj = OrthoCamCalPair(p,nPat,nPts,worldPts,imPts,calPatFPNs)
      obj.tblInt = OrthoCam.summarizeIntrinsicsStro(p,nPat);
      [~,~,~,~,~,~,~,~,~,~,~,~,...
       obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,rvecs2thrun,tvecs2thrun] = ...
         OrthoCam.unpackParamsStro(p,nPat);
      obj.rvecs = [0 0 0;rvecs2thrun];
      obj.tvecs = [0 0 0;tvecs2thrun];
      
      obj.calNumPatterns = nPat;
      obj.calNumPoints = nPts;
      szassert(worldPts,[3 nPts]);
      szassert(imPts,[2 nPts nPat 2]);
      szassert(calPatFPNs,[nPat 2]);
      obj.calWorldPoints = worldPts;
      obj.calImPoints = imPts;
      obj.calPatternFPNs = calPatFPNs;
      [obj.optCtr1,~,~,~,obj.ijkCamWorld1] = ...
        OrthoCam.opticalCenter(vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec1),obj.t2vec1);
      [obj.optCtr2,~,~,~,obj.ijkCamWorld2] = ...
        OrthoCam.opticalCenter(vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2),obj.t2vec2);
      obj.optCtr1(end+1) = 0;
      obj.optCtr2(end+1) = 0;
      obj.calTS = now;
    end
    
    function hFig = viewExtrinsics(obj)
      hFig = OrthoCam.viewExtrinsics(obj.calWorldPoints,...
        obj.rvecs,obj.tvecs,obj.r2vec1,obj.t2vec1,obj.r2vec2,obj.t2vec2,...
        'cam1info',struct('optCtr',obj.optCtr1,'n',-obj.ijkCamWorld1(:,3),'ijkCamWorld',obj.ijkCamWorld1),...
        'cam2info',struct('optCtr',obj.optCtr2,'n',-obj.ijkCamWorld2(:,3),'ijkCamWorld',obj.ijkCamWorld2));
    end
    
    function xformWorldSys(obj,R)
      % Transform WorldCoords by rotation (origin unchanged)
      % R: [3x3]. x_newWorldSys = R*x_oldWorldSys
            
      szassert(R,[3 3]);
      for iPat=1:obj.calNumPatterns
        R0 = vision.internal.calibration.rodriguesVectorToMatrix(obj.rvecs(iPat,:)');
        R1 = R*R0;
        obj.rvecs(iPat,:) = vision.internal.calibration.rodriguesMatrixToVector(R1);
        
        t0 = obj.tvecs(iPat,:)';
        t1 = R*t0;
        obj.tvecs(iPat,:) = t1;
      end
      
      obj.r2vec1 = lclHelpR(obj.r2vec1,R);
      obj.r2vec2 = lclHelpR(obj.r2vec2,R);
      
      obj.optCtr1 = R*obj.optCtr1;
      obj.optCtr2 = R*obj.optCtr2;
      obj.ijkCamWorld1 = R*obj.ijkCamWorld1;
      obj.ijkCamWorld2 = R*obj.ijkCamWorld2;
    end
    
    % Coordsys Notes
    % X=[x;y;z] 3D worldsys coords
    % pq=[p;q] Normalized coords; undistorted, unmagnified x-y deviations
    % from optical axis along camera x-y axes
    % uv=[u;v] Image coords; (col,row) pixel coords on image
    
    function [X,d,uvreL,uvreR] = stereoTriangulate(obj,uvL,uvR)
      % [X,d,uvreL,uvreR] = stereoTriangulate(obj,uvL,uvR)
      % Stereo triangulation
      %
      % uvL, uvR: [2xN] x-y image coords
      % 
      % X: [3xN]: reconstructed world coords
      % d: [1xN]: error/discrepancy in closest approach. d=0 indicates
      %   apparently "perfect" reconstruction where epipolar rays meet
      % uvreL, uvreR: [2xN]: reprojected x-y image coords
      
      pqL = obj.projected2normalized(uvL,1);
      pqR = obj.projected2normalized(uvR,2);
      
      n = size(pqL,2);
      szassert(pqL,[2 n]);
      szassert(pqR,[2 n]);
      
      OL = obj.optCtr1 + pqL(1,:).*obj.ijkCamWorld1(:,1) ...
                       + pqL(2,:).*obj.ijkCamWorld1(:,2);
      OR = obj.optCtr2 + pqR(1,:).*obj.ijkCamWorld2(:,1) ...
                       + pqR(2,:).*obj.ijkCamWorld2(:,2);
      szassert(OL,[3 n]);
      szassert(OR,[3 n]);
      X = nan(3,n);
      d = nan(1,n);      
      n1 = -obj.ijkCamWorld1(:,3);
      n2 = -obj.ijkCamWorld2(:,3);
      for i=1:n
        [P,Q,d(i)] = CalibratedRig.stereoTriangulateRays(OL(:,i),n1,OR(:,i),n2);
        X(:,i) = (P+Q)/2;
      end
      
      uvreL = obj.project(X,1);
      uvreR = obj.project(X,2);
    end
    
    function uv = project(obj,X,icam)
      % uv = project(obj,X,icam)
      % Project 3D world coords to camera image
      % 
      % X: [3xn] world coords
      % icam: camera index, 1/2 for left/right resp
      %
      % uv: [2xn] x- and y-pixel coords (NOT row/col) in camera icam
      
      assert(size(X,1)==3);
      assert(icam==1 || icam==2);
      
      switch icam
        case 1
          R2 = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec1);
          t2 = obj.t2vec1;
        case 2
          R2 = vision.internal.calibration.rodriguesVectorToMatrix(obj.r2vec2);
          t2 = obj.t2vec2;
      end
      int = obj.tblInt(icam,:);
      uv = OrthoCam.project(X,R2,t2,int.k1,int.k2,int.mx,int.my,int.u0,int.v0);
    end
    
    function pq = projected2normalized(obj,uv,icam)
      % uv: [2xn]
      % icam: camera index
      %
      % pq: [2xn]
      
      assert(isnumeric(icam) && (icam==1 || icam==2));
      ints = obj.tblInt(icam,:);
      pq = OrthoCam.projected2normalized(ints.mx,ints.my,ints.u0,ints.v0,...
        ints.k1,ints.k2,uv);
    end
    
  end
  
  methods %CalRig
    
    function [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,uv1,iViewEpi)
      % [xEPL,yEPL] = computeEpiPolarLine(obj,iView1,xy1,iViewEpi)
      % 
      % iView1: either 1 (L) or 2 (R)
      % uv1: [2] x-y image coords
      % iViewEpi: either 1 (L) or 2 (R)
      %
      % xEPL, yEPL: [nx1] each; points in epipolar line

      assert(iView1==1 || iView1==2);
      assert(numel(uv1)==2);
      assert(iViewEpi==1 || iViewEpi==2);
      
      pq1 = obj.projected2normalized(uv1(:),iView1);
      
      switch iView1
        case 1
          optCtr = obj.optCtr1;
          ijkCam = obj.ijkCamWorld1;
        case 2
          optCtr = obj.optCtr2;
          ijkCam = obj.ijkCamWorld2;
      end
      
      O1 = optCtr + pq1(1)*ijkCam(:,1) + pq1(2)*ijkCam(:,2);
      szassert(O1,[3 1]);
      MAXS = 7; % mm
      DS = .05; % mm
      s = -MAXS:DS:MAXS;
      XEPL = O1 + s.*ijkCam(:,3);
      
      uvEPL = obj.project(XEPL,iViewEpi);
      rc = uvEPL([2 1],:)';
      rcCrop = obj.cropLines(rc,iViewEpi);
      xEPL = rcCrop(:,2);
      yEPL = rcCrop(:,1);
    end

    function [u_p,v_p,w_p] = reconstruct2d(obj,x,y,iView)
      assert(isequal(size(x),size(y)));
      assert(isvector(x));
      
      uv = [x(:) y(:)]';
      pq = obj.projected2normalized(uv,iView); % [2xn]
      
      assert(false,'TODO');
%       
%       % each col of pq is a normalized pt. The World-line corresponding to
%       % pq(:,i) is the camera axis 
%       szassert(pq,[2 n]);
%       
%       
%       n = numel(x);
%       dlt = obj.getDLT(iView);      
%       u_p = nan(n,2);
%       v_p = nan(n,2);
%       w_p = nan(n,2);
%       for i=1:n
%         [u_p(i,:),v_p(i,:),w_p(i,:)] = dlt_2D_to_3D(dlt,x(i),y(i));
%       end
    end
        
    function [x,y] = project3d(obj,u,v,w,iView)
      assert(isequal(size(u),size(v),size(w)));
      
      X = [u(:)';v(:)';w(:)'];
      uv = obj.project(X,iView);
      x = uv(1,:)';
      y = uv(2,:)';
    end
    
  end
  
end

function r2vecnew = lclHelpR(r2vec,R)
Rworld2cam = vision.internal.calibration.rodriguesVectorToMatrix(r2vec);
Rworld2camNew = Rworld2cam*R';
r2vecnew = vision.internal.calibration.rodriguesMatrixToVector(Rworld2camNew);
end