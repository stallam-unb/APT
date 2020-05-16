function s = saveJSONfile(data, jsonFileName)
% saves the values in the structure 'data' to a file in JSON format.
%
% Example:
%     data.name = 'chair';
%     data.color = 'pink';
%     data.metrics.height = 0.3;
%     data.metrics.width = 1.3;
%     saveJSONfile(data, 'out.json');
%
% Output 'out.json':
% {
% 	"name" : "chair",
% 	"color" : "pink",
% 	"metrics" : {
% 		"height" : 0.3,
% 		"width" : 1.3
% 		}
% 	}
%
% original version from 
% Lior Kirsch (2020). 
% Structure to JSON 
% (https://www.mathworks.com/matlabcentral/fileexchange/50965-structure-to-json)
% MATLAB Central File Exchange. Retrieved May 14, 2020. 
%
% modified by Kristin Branson
s = '';
if exist('jsonFileName','var'),
  fid = fopen(jsonFileName,'w');
  writefun = @(varargin) fprintf(fid,varargin{:});
else
  fid = [];
  writefun = @(varargin) addToString(sprintf(varargin{:}));
end

if iscell(data),
  writefun('[\n');
  tabs = sprintf('\t');
  for j = 1:numel(data),
    writeElement(writefun,data{j},tabs);
    if j < numel(data),
      writefun(',\n%s',tabs);
    end
  end
  writefun(']\n');
else
  writeElement(writefun, data,'');
end
if exist('jsonFileName','var'),
  fprintf(fid,'\n');
  fclose(fid);
end

  function writeElement(writefun, data,tabs)
    namesOfFields = fieldnames(data);
    numFields = length(namesOfFields);
    tabs = sprintf('%s\t',tabs);
    writefun('{\n%s',tabs);
    
    for i = 1:numFields - 1
      currentField = namesOfFields{i};
      currentElementValue = eval(sprintf('data.%s',currentField));
      writeSingleElement(writefun, currentField,currentElementValue,tabs);
      writefun(',\n%s',tabs);
    end
    if isempty(i)
      i=1;
    else
      i=i+1;
    end
    
    
    currentField = namesOfFields{i};
    currentElementValue = eval(sprintf('data.%s',currentField));
    writeSingleElement(writefun, currentField,currentElementValue,tabs);
    writefun('\n%s}',tabs);
  end

  function writeSingleElement(writefun, currentField,currentElementValue,tabs)
    
    % if this is an array and not a string then iterate on every
    % element, if this is a single element write it
    if length(currentElementValue) > 1 && ~ischar(currentElementValue)
      writefun('"%s" : [',currentField);
      if isnumeric(currentElementValue),
        scurr = sprintf('%g,',currentElementValue);
        scurr = scurr(1:end-1);
        writefun(scurr);
      else
        writefun('\n%s',tabs);
        for m = 1:length(currentElementValue)-1
          cv = currentElementValue(m);
          if iscell(cv),
            cv = cv{1};
          end
          writeElement(writefun,cv,tabs);
          writefun(',\n%s',tabs);
        end
        if isempty(m)
          m=1;
        else
          m=m+1;
        end
        cv = currentElementValue(m);
        if iscell(cv),
          cv = cv{1};
        end
        writeElement(writefun, cv,tabs);
        writefun('\n%s',tabs);
      end
      
      writefun(']\n%s',tabs);
    elseif isstruct(currentElementValue)
      writefun('"%s" : ',currentField);
      writeElement(writefun, currentElementValue,tabs);
    elseif isnumeric(currentElementValue)
      if isempty(currentElementValue),
        writefun('"%s" : []' , currentField);
      else
        writefun('"%s" : %g' , currentField,currentElementValue);
      end
%     elseif isempty(currentElementValue)
%       writefun('"%s" : null' , currentField,currentElementValue);
    else %ischar or something else ...
      writefun('"%s" : "%s"' , currentField,currentElementValue);
    end
  end

  function addToString(s1)
    s = [s,s1];
  end

end
