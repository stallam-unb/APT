classdef TrainMonitorViz < handle
  properties
    hfig % scalar fig
    haxs % [2] axis handle, viz training loss, dist
    hannlastupdated % [1] textbox/annotation handle
    hline % [nviewx2] line handle, one loss curve per view
    hlinekill % [nviewx2] line handle, killed marker per view
    
    resLast % last training json contents received
  end
  
  methods
    function obj = TrainMonitorViz(nview)
      obj.hfig = figure('Visible','on');
      obj.haxs = reshape(createsubplots(2,1,.1),2,1);
      arrayfun(@(x)grid(x,'on'),obj.haxs);
      arrayfun(@(x)hold(x,'on'),obj.haxs);
      title(obj.haxs(1),'Training Monitor','fontweight','bold');
      xlabel(obj.haxs(2),'Iteration');
      ylabel(obj.haxs(1),'Loss');
      ylabel(obj.haxs(2),'Dist');
      linkaxes(obj.haxs,'x');
            
      obj.hannlastupdated = TrainMonitorViz.createAnnUpdate(obj.haxs(1));
      
      clrs = lines(nview);
      h = gobjects(nview,2);
      hkill = gobjects(nview,2);
      for ivw=1:nview
        for j=1:2
          h(ivw,j) = plot(obj.haxs(j),nan,nan,'.-','color',clrs(ivw,:));
          hkill(ivw,j) = plot(obj.haxs(j),nan,nan,'rx','markersize',12,'linewidth',2);
        end
      end
      viewstrs = arrayfun(@(x)sprintf('view%d',x),(1:nview)','uni',0);
      legend(obj.haxs(2),h(:,1),viewstrs);
      obj.hline = h;
      obj.hlinekill = hkill;
      obj.resLast = [];
    end
    function delete(obj)
      deleteValidHandles(obj.hfig);
      obj.hfig = [];
      obj.haxs = [];
    end
    function resultsReceived(obj,sRes)
      % Callback executed when new result received from training monitor BG
      % worker
      %
      % trnComplete: scalar logical, true when all views done
      
      res = sRes.result;
      tfAnyUpdate = false;
      
      h = obj.hline;
      for ivw=1:numel(res)
        if res(ivw).pollsuccess
          if res(ivw).jsonPresent && res(ivw).tfUpdate
            contents = res(ivw).contents;
            set(h(ivw,1),'XData',contents.step,'YData',contents.train_loss);
            set(h(ivw,2),'XData',contents.step,'YData',contents.train_dist);
            lclAutoAxisWithYLim0(obj.haxs(1));
            lclAutoAxisWithYLim0(obj.haxs(2));
            tfAnyUpdate = true;
          end

          if res(ivw).killFileExists && res(ivw).jsonPresent
            contents = res(ivw).contents;
            hkill = obj.hlinekill;
            % hmm really want to mark the last 2k interval when model is
            % actually saved
            set(hkill(ivw,1),'XData',contents.step(end),'YData',contents.train_loss(end));
            set(hkill(ivw,2),'XData',contents.step(end),'YData',contents.train_dist(end));
          end
        
          if res(ivw).trainComplete
            contents = res(ivw).contents;
            if ~isempty(contents)
              hkill = obj.hlinekill;
              % re-use kill marker 
              set(hkill(ivw,1),'XData',contents.step(end),'YData',contents.train_loss(end),...
                'color',[0 0.5 0],'marker','o');
              set(hkill(ivw,2),'XData',contents.step(end),'YData',contents.train_dist(end),...
                'color',[0 0.5 0],'marker','o');
            end
          end
        end
      end
      
      obj.updateAnn([res.pollsuccess]);
      
      if isempty(obj.resLast) || tfAnyUpdate
        obj.resLast = res;
      end
      
%           
%           
%         fprintf(1,'View%d: jsonPresent: %d. ',ivw,res(ivw).jsonPresent);
%         if res(ivw).tfUpdate
%           fprintf(1,'New training iter: %d.\n',res(ivw).lastTrnIter);
%         elseif res(ivw).jsonPresent
%           fprintf(1,'No update, still on iter %d.\n',res(ivw).lastTrnIter);
%         else
%           fprintf(1,'\n');
%         end
    end
    function updateAnn(obj,pollsuccess)
      % pollsuccess: [nview] logical
      % pollts: [nview] timestamps
      
      str = sprintf('last updated: %s',datestr(now,'HH:MM:SS PM'));
      hAnn = obj.hannlastupdated;
      hAnn.String = str;
      
      tfsucc = all(pollsuccess);
      if all(tfsucc)
        hAnn.Color = [0 0.5 0];
      else
        hAnn.Color = [1 0 0];
      end
      
      ax = obj.haxs(1);
      hAnn.Position(1) = ax.Position(1)+ax.Position(3)-hAnn.Position(3);
      hAnn.Position(2) = ax.Position(2)+ax.Position(4)-hAnn.Position(4);
    end
  end
  
  methods (Static)
    function hAnn = createAnnUpdate(ax)
      hfig = ax.Parent;
      ax.Units = 'normalized';
      hfig.Units = 'normalized';
      str = sprintf('last updated: %s',datestr(now,'HH:MM:SS PM'));
      hAnn = annotation(hfig,'textbox',[1 1 0.1 0.1],...
        'String',str,'FitBoxToText','on','EdgeColor',[1 1 1],...
        'FontAngle','italic','HorizontalAlignment','right');
      drawnow;
      hAnn.Position(1) = ax.Position(1)+ax.Position(3)-hAnn.Position(3);
      hAnn.Position(2) = ax.Position(2)+ax.Position(4)-hAnn.Position(4);
   end
  end
  
end

function lclAutoAxisWithYLim0(ax)
axis(ax,'auto');
yl = ylim(ax);
yl(1) = 0;
ylim(ax,yl);
end