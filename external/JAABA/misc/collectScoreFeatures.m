function scoreFeatures= ...
  collectScoreFeatures(fileNameList, ...
                       timeStampList, ...
                       scoreFileBaseNameList)
                        
% Collects the three fields that make up the scoreFeatures structure into 
% a structure with proper field names.  1st and 3rd args are column cell
% arrays of strings, 2nd arg is a column array.

scoreFeatures=struct('classifierfile',fileNameList, ...
                     'ts',num2cell(timeStampList), ...
                     'scorefilename',scoreFileBaseNameList);

end
