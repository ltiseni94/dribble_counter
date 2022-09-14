function fileList = getAllFiles(dirName,varargin)

% FUNCTION fileList = getAllFiles(dirName,opt)
% function which give the list of all the files in a directory and the
% subdirectories if opt == 1
%
%---INPUTS:---------------------------------------------------------------%
%   dirName    : a string of the name of a directory
%   opt        : 0 for listing the files only in the given directory
%                1 for listing the files in the given directory and
%                subdirectories.
%
%---OUTPUTS:--------------------------------------------------------------%
%   fileList    : a cell array of file names.
%
% Last edited : Sandrine Muller 13/07/26.
% Modified by: Michael Lassi

    %% Input verification and initialization:
    if ~ischar(dirName) % check for the nam of the directory
        error('The name for the directory should be a string.');
    end
    if length(varargin) == 0
        SubDirYESNO = 0;
        extension ='';
    elseif length(varargin) >= 1
        if length(varargin{1}) > 1
            error('The option should not be a vector.');
        elseif varargin{1} ~= 1 && varargin{1} ~= 0
            error('The option should be 1 or 0');
        else
            SubDirYESNO = varargin{1};
        end
        extension ='';
    if length(varargin) == 2 
       extension = varargin{2};
    end
   
    %% Treat for current directory:
    dirData = dir(dirName);      % Get the data for the current directory
    dirIndex = [dirData.isdir];  % Find the index for directories
    fileList = {dirData(~dirIndex).name}';  % Get a list of the files
    fileList = fileList(endsWith(fileList,extension));
    
    if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  % Prepend path to files
                       fileList,'UniformOutput',false);
    end
    
    %% Treat by recursivity for subdirectories if necessary: 
    if SubDirYESNO == 1
        subDirs = {dirData(dirIndex).name};  % Get a list of the subdirectories
        validIndex = ~ismember(subDirs,{'.','..'});  % Find index of subdirectories that are not '.' or '..'
        
        for iDir = find(validIndex)                  % Loop over valid subdirectories
            nextDir = fullfile(dirName,subDirs{iDir});    % Get the subdirectory path
            fileList = [fileList; getAllFiles(nextDir,1,extension)];  % Recursively call getAllFiles
        end
      
    end

end