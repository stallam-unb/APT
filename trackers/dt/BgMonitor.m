classdef BgMonitor < handle
  % BGMonitor
  %
  % A BGMonitor is:
  % 1. A BGClient/BGWorker pair comprising a client, bg worker working
  % asynchronously calling meths on a BgWorkerObj, and a 2-way comm 
  % pipeline.
  %   - The initted BgWorkerObj knows how to poll the state of the process. For
  %     debugging/testing this can be done from the client machine.
  % 2. A client-side MonitorViz object that visualizes 
  % progress sent back from the BGWorker
  % 3. Custom actions performed when process is complete
  %
  % BGMonitor does NOT know how to spawn process jobs but will know
  % how to (attempt to) kill them. For debugging, you can manually spawn 
  % jobs and monitor them with BgMonitor.
  %
  % BGMonitor does NOT know how to probe the detailed state of the
  % process eg on disk. That is BGWorkerObj's domain.
  %
  % So BGMonitor is a connector/manager obj that runs the worker 
  % (knows how to poll the filesystem in detail) in the background and 
  % connects it with a Monitor.
  %
  % See also prepare() method comments for related info.
  
  properties
    bgContCallInterval = 30; %secs
    
    bgClientObj
    bgWorkerObj % scalar "detached" object that is deep-copied onto
    % workers. Note, this is not the BGWorker obj itself
    monitorObj % object with resultsreceived() method
    cbkComplete = []; % fcnhandle with sig cbk(res), called when operation complete
    processName = 'process';
  end
  properties (Dependent)
    prepared
    isRunning
  end
  
  events
    bgStart
    bgEnd    
  end
  
  methods
    function v = get.prepared(obj)
      v = ~isempty(obj.bgClientObj);
    end
    function v = get.isRunning(obj)
      bgc = obj.bgClientObj;
      v = ~isempty(bgc) && bgc.isRunning;
    end
  end
  
  methods
    
    function obj = BgMonitor
      obj.reset();
    end
    
    function delete(obj)
      obj.reset();
    end
    
    function reset(obj)
      % Reset BG Monitor state
      %
      % - TODO Note, when you change eg params, u need to call this. etc etc.
      % Any mutation that alters PP, train/track on the BG worker...
      
      if obj.isRunning()
        obj.notify('bgEnd');
      end
      
      if ~isempty(obj.bgClientObj)
        delete(obj.bgClientObj);
      end
      obj.bgClientObj = [];
      
      if ~isempty(obj.bgWorkerObj)
        delete(obj.bgWorkerObj)
      end
      obj.bgWorkerObj = [];
      
      obj.cbkComplete = [];
      
      if ~isempty(obj.monitorObj)
        delete(obj.monitorObj);
      end
      obj.monitorObj = [];
    end
    
    function prepare(obj,monVizObj,bgWorkerObj,cbkComplete)
      % bgWorkerObj knows how to poll the state of the process. 
      % monVizObj knows how to vizualize this state. 
      % bgResultsReceivedHook performs custom actions after receiving
      % an update from bgWorkerObj. 
      %
      % bgWorkerObj/monVizObj should be mix+matchable as bgWorkerObj 
      % should send a core set of 'standard' metrics that monVizObj can
      % use.
      %
      % bgWorkerObj matches 1-1 with the concrete BgMonitor and its 
      % bgResultsReceivedHook method. These work in concert and the 
      % custom actions taken by bgResultsReceivedHook depends on custom 
      % info supplied by bgWorkerObj.
      
      obj.reset();
      
      [tfEFE,errFile] = bgWorkerObj.errFileExists;
      if tfEFE
        error('Error file ''%s'' exists.',errFile);
      end
      
      cbkResult = @obj.bgResultsReceived;

      bgc = BGClient;
      fprintf(1,'Configuring background worker...\n');
      bgc.configure(cbkResult,bgWorkerObj,'compute');
      
      obj.bgClientObj = bgc;
      obj.bgWorkerObj = bgWorkerObj;
      obj.monitorObj = monVizObj;
      if exist('cbkComplete','var'),
        obj.cbkComplete = cbkComplete;
      end
      
      %obj.prepareHook(monVizObj,bgWorkerObj);
    end
    
    function start(obj)
      assert(obj.prepared);
      bgc = obj.bgClientObj;
      bgc.startWorker('workerContinuous',true,...
        'continuousCallInterval',obj.bgContCallInterval);
      obj.notify('bgStart');
    end
    
    function bgResultsReceived(obj,sRes)
      obj.monitorObj.resultsReceived(sRes);
      obj.bgResultsReceivedHook(sRes);
    end
    
    function tfpollsucc = getPollSuccess(obj,sRes)
      if isfield(sRes.result,'pollsuccess'),
        tfpollsucc = [sRes.result.pollsuccess];
      else
        tfpollsucc = true(1,numel(sRes.result));
      end
    end

    function killOccurred = getKillOccurred(obj,sRes)
      killOccurred = [sRes.result.killFileExists];
    end

    function errOccurred = getErrOccurred(obj,sRes)
      errOccurred = [sRes.result.errFileExists];
    end
    
    function errFile = getErrFile(obj,sRes)
      errFile = sRes.result(1).errFile;
    end
    function logFile = getLogFile(obj,sRes,i)
      logFile = sRes.result(i).logFile;
    end
    function logFileErrLikely = getLogFileErrLikely(obj,sRes)
      logFileErrLikely = [sRes.result.logFileErrLikely];
    end

    function tfComplete = isComplete(obj,sRes)
      tfComplete = [sRes.result.tfComplete];
    end

    
    function bgResultsReceivedHook(obj,sRes)
      % current pattern is, this meth only handles things which stop the
      % process. everything else handled by monitor
      
      tfpollsucc = obj.getPollSuccess(sRes);
      
      killOccurred = any(tfpollsucc & obj.getKillOccurred(sRes));
      if killOccurred
        obj.stop();        
        fprintf(1,'Process killed!\n');
        % monitor plot stays up; reset not called etc
      end
      
      errOccurred = any(tfpollsucc & obj.getErrOccurred(sRes));
      if errOccurred
        obj.stop();

        fprintf(1,'Error occurred during %s:\n',obj.processName);
        errFile = obj.getErrFile(sRes); % currently, errFiles same for all views
        fprintf(1,'\n### %s\n\n',errFile);
        errContents = obj.bgWorkerObj.fileContents(errFile);
        disp(errContents);
        fprintf(1,'\n\n. You may need to manually kill any running DeepLearning process.\n');
        
        % monitor plot stays up; reset not called etc
      end
      logFileErrLikely = obj.getLogFileErrLikely(sRes);
      for i=1:numel(sRes.result)
        if tfpollsucc(i) && logFileErrLikely(i),
          obj.stop();
          
          fprintf(1,'Error occurred during %s:\n',obj.processName);
          errFile = obj.getLogFile(sRes,i);
          fprintf(1,'\n### %s\n\n',errFile);
          errContents = obj.bgWorkerObj.fileContents(errFile);
          disp(errContents);
          fprintf(1,'\n\n. You may need to manually kill any running %s process.\n',obj.processName);
          
          % monitor plot stays up; bgReset not called etc
        end
      end
      
      tfComplete = all(tfpollsucc & obj.isComplete(sRes));
      if tfComplete
        obj.stop();
        % % monitor plot stays up; reset not called etc
        fprintf('%s complete at %s.\n',obj.processName,datestr(now));
        
        if ~isempty(obj.cbkComplete),
          obj.cbkComplete(sRes.result);
        end
      end
    end
    
    function stop(obj)
      bgc = obj.bgClientObj;
      bgc.stopWorkerHard();
      obj.notify('bgEnd');
    end
    
  end
  
%   methods (Abstract)
%     prepareHook(obj,monVizObj,bgWorkerObj)    
%   end
end