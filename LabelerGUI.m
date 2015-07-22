function varargout = LabelerGUI(varargin)
% LARVALABELER MATLAB code for LarvaLabeler.fig
%      LARVALABELER, by itself, creates a new LARVALABELER or raises the existing
%      singleton*.
%
%      H = LARVALABELER returns the handle to a new LARVALABELER or the handle to
%      the existing singleton*.
%
%      LARVALABELER('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in LARVALABELER.M with the given input arguments.
%
%      LARVALABELER('Property','Value',...) creates a new LARVALABELER or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before LabelerGUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to LabelerGUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help LarvaLabeler

% Last Modified by GUIDE v2.5 22-Jul-2015 08:45:53

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @LabelerGUI_OpeningFcn, ...
                   'gui_OutputFcn',  @LabelerGUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1}) && exist(varargin{1}),
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT

% --- Executes just before LarvaLabeler is made visible.
% 
function LabelerGUI_OpeningFcn(hObject, eventdata, handles, varargin)

% Choose default command line output for LarvaLabeler
% global LARVALABELERLASTMOVIEPATH;
handles.output = hObject;

% parse inputs
handles.labelerObj = varargin{1};
varargin = varargin(2:end);
[handles.moviefile,handles.template,handles.trxfile] = myparse(varargin,'moviefile','',...
  'template',[],'trxfile','');

% select video to label
% if isempty(handles.moviefile)  
%   handles.moviefile = SelectVideo();
% else
% %   LARVALABELERLASTMOVIEPATH = fileparts(handles.moviefile);
% end

handles.npoints = 0;
handles.keypressmode = '';
handles.motionobj = [];

handles = InitializeGUI(handles);

% load initial template
if ~isempty(handles.template) && ischar(handles.template),
  load(handles.template,'template');
  handles.template = template;
end

guidata(hObject, handles);

% UIWAIT makes LarvaLabeler wait for user response (see UIRESUME)
% uiwait(handles.figure);



function handles = InitializeGUI(handles)

% set(handles.figure,'ToolBar','figure');

handles.image_curr = imagesc(0,'Parent',handles.axes_curr);
set(handles.image_curr,'hittest','off');
axis(handles.axes_curr,'image');
axisoff(handles.axes_curr);
hold(handles.axes_curr,'on');
set(handles.axes_curr,'Color',[0 0 0]); 
%colormap(handles.figure,fire_colormap(2^8));
colormap(handles.figure,gray);

handles.image_prev = imagesc(0,'Parent',handles.axes_prev);
set(handles.image_prev,'hittest','off');
axis(handles.axes_prev,'image');
axisoff(handles.axes_prev);
hold(handles.axes_prev,'on');
handles.posprev = [];

linkaxes([handles.axes_prev,handles.axes_curr]);

fcn = get(handles.slider_frame,'Callback');
handles.hslider_listener = addlistener(handles.slider_frame,'ContinuousValueChange',fcn);
set(handles.slider_frame,'Callback','');

handles.hpoly = [];
handles.pointselected = [];
handles.htext = [];
handles.dt2p = 5;

handles.f = 1;
handles.maxv = inf;
handles.minv = 0;
set(handles.output,'Toolbar','figure');

colnames = handles.labelerObj.TBLTRX_STATIC_COLSTBL;
set(handles.tblTrx,'ColumnName',colnames,'Data',cell(0,numel(colnames)));      
      
colnames = handles.labelerObj.TBLFRAMES_COLS;
set(handles.tblFrames,'ColumnName',colnames,'Data',cell(0,numel(colnames)));

%handles = InitializeVideo(handles);

%handles = UpdateFrame(handles);

hchil = findall(handles.figure,'-property','KeyPressFcn');
handles.keypressfcn = get(handles.figure,'KeyPressFcn');
set(hchil,'KeyPressFcn',handles.keypressfcn);

function slider_frame_Callback(hObject,~,handles)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
lObj = handles.labelerObj;
v = get(hObject,'Value');
f = round(1 + v * (lObj.nframes - 1));
lObj.setFrame(f);

function slider_frame_CreateFcn(hObject,~,~)
% Hint: slider controls usually have a light gray background.
if isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor',[.9 .9 .9]);
end

function edit_frame_Callback(hObject,~,handles)
lObj = handles.labelerObj;
f = str2double(get(hObject,'String'));
if isnan(f)
  set(hObject,'String',num2str(lObj.currFrame));
  return;
end
f = min(max(1,round(f)),lObj.nframes);
set(hObject,'String',num2str(f));
if f ~= lObj.currFrame
  lObj.setFrame(f)
end 
  
function edit_frame_CreateFcn(hObject,~,~)
if ispc && isequal(get(hObject,'BackgroundColor'), ...
                   get(0,'defaultUicontrolBackgroundColor'))
  set(hObject,'BackgroundColor','white');
end

function pbClear_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.labelMode1Label();

function tbAccept_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
switch lObj.lbl1_state
  case SequentialModeState.ADJUST
    lObj.labelMode1Accept(true);    
  case SequentialModeState.ACCEPTED
    lObj.labelMode1Adjust();
  otherwise
    assert(false);
end

function tblTrx_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to tblTrx (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
lObj = handles.labelerObj;
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  lObj.setTarget(row);
end

function tblFrames_CellSelectionCallback(hObject, eventdata, handles)
% hObject    handle to tblFrames (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
lObj = handles.labelerObj;
row = eventdata.Indices;
if ~isempty(row)
  row = row(1);
  dat = get(hObject,'Data');
  lObj.setFrame(dat{row,1});
end

function sldZoom_Callback(hObject, eventdata, handles)
% Hints: get(hObject,'Value') returns position of slider
%        get(hObject,'Min') and get(hObject,'Max') to determine range of slider
lObj = handles.labelerObj;
zoomFac = get(hObject,'Value');
lObj.videoZoomFac(zoomFac);

function pbResetZoom_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.videoResetView();

%% menu
function menu_file_save_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
lObj.saveLblFile();

function tfcontinue = hlpSave(labelerObj)
tfcontinue = true;
if labelerObj.hasMovie
  res = questdlg('Save before closing current video?');
  switch res
    case 'No'
      % none
    case 'Cancel'
      tfcontinue = false;
    case 'Yes'
      assert(false,'XXX TODO');
      handles = SaveState(handles);
  end
end

function menu_file_openmovie_Callback(hObject,~,handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  lObj.loadMovie();
  % XXX add me somewhere
  % handles.templatecolors = jet(handles.npoints);
end

function menu_file_openmovietrx_Callback(hObject, eventdata, handles)
lObj = handles.labelerObj;
if hlpSave(lObj)
  lObj.loadMovie([],[]);
end


% % % below is untouched % % % ---------------------


% % tbLocked_Callback
% handles.islocked(handles.f,handles.animal) = get(hObject,'Value');
% if handles.islocked(handles.f,handles.animal),
%   handles.labeledpos(:,1,handles.f,handles.animal) = cell2mat(get(handles.hpoly,'XData'));
%   handles.labeledpos(:,2,handles.f,handles.animal) = cell2mat(get(handles.hpoly,'YData'));
% end

function handles = SetTemplate(handles)
  
uiwait(msgbox('Click to create template points. First, click to create each point. Then you can drag points around. Hit escape when done.'));

handles.hpoly = [];
handles.htext = [];
handles.keypressmode = 'settemplate';
handles.template = nan(0,2);

axes(handles.axes_curr);

while true,

  keydown = waitforbuttonpress;
  if get(0,'CurrentFigure') ~= handles.figure,
    continue;
  end
  if keydown == 0 && strcmpi(get(handles.figure,'SelectionType'),'normal'),
    tmp = get(handles.axes_curr,'CurrentPoint');
    x = tmp(1,1);
    y = tmp(1,2);
    handles.hpoly(end+1) = plot(handles.axes_curr,x,y,'w+','MarkerSize',20,'LineWidth',3);%,...
      %'KeyPressFcn',handles.keypressfcn);
    handles.htext(end+1) = text(x+handles.dt2p,y,num2str(numel(handles.hpoly)),'Parent',handles.axes_curr);%,...
    handles.template(end+1,:) = [x,y];
  elseif keydown == 1 && double(get(handles.figure,'CurrentCharacter')) == 27,
    break;
  end
  
end

if ~isempty(handles.trx)
  trxndx = handles.f + 1 - handles.trx(handles.animal).firstframe;
  handles.templateloc = [handles.trx(handles.animal).x(trxndx) handles.trx(handles.animal).y(trxndx)];
  handles.templatetheta = handles.trx(handles.animal).theta(trxndx);
end

handles.npoints = numel(handles.hpoly);
handles.templatecolors = jet(handles.npoints);%*.5+.5;
for i = 1:handles.npoints,

  set(handles.hpoly(i),'Color',handles.templatecolors(i,:),...
    'ButtonDownFcn',@(hObject,eventdata) PointButtonDownCallback(hObject,eventdata,handles.figure,i));
  %addNewPositionCallback(handles.hpoly(i),@(pos) UpdateLabels(pos,handles.figure,i));
  set(handles.htext(i),'Color',handles.templatecolors(i,:));
  
end

function PointButtonDownCallback(hObject,eventdata,hfig,i)

handles = guidata(hfig);
handles.motionobj = i;
handles.didmovepoint = false;
guidata(hObject,handles);

function UpdateLabels(pos,hfig,i,handles)

if nargin < 4,
  handles = guidata(hfig);
end
handles.labeledpos(i,:,handles.f,handles.animal) = pos;
guidata(hfig,handles);

function handles = InitializeVideo(handles)
% initVideo


function handles = UpdateFrame(handles)
handles.labelerObj.setFrame(handles.f);

% --- Outputs from this function are returned to the command line.
function varargout = LabelerGUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;





% --------------------------------------------------------------------
function menu_file_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)



function handles = SaveState(handles)

global LARVALABELERSAVEFILE;

savedata = struct;
savedata.labeledpos = handles.labeledpos;
savedata.islocked = handles.islocked;
savedata.moviefile = handles.moviefile;
savedata.template = handles.template;
savedata.npoints = handles.npoints;
savedata.f = handles.f;
savedata.minv = handles.minv;
savedata.maxv = handles.maxv;
savedata.templateloc = handles.templateloc;
savedata.templatetheta = handles.templatetheta;

if ~isfield(handles,'savefile'),
  handles.savefile = '';
end

[f,p] = uiputfile('*.mat','Save labels to file',handles.savefile);
if ~ischar(f),
  return;
end
handles.savefile = fullfile(p,f);
LARVALABELERSAVEFILE = handles.savefile;

save(handles.savefile,'-struct','savedata');

% --------------------------------------------------------------------
function menu_file_quit_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_quit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

CloseGUI(handles);


% --------------------------------------------------------------------
function menu_setup_Callback(hObject, eventdata, handles)
% hObject    handle to menu_setup (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_setup_settemplate_Callback(hObject, eventdata, handles)
% hObject    handle to menu_setup_settemplate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if ~isempty(handles.template),
  
  res = questdlg('Changing template will result in all labels being cleared. Save before doing this?');
  if strcmpi(res,'Cancel'),
    return;
  elseif strcmpi(res,'Yes'),
    handles = SaveState(handles);
  end

  % delete the current template
  for i = 1:handles.npoints,
    try %#ok<TRYNC>
      delete(handles.hpoly(i));
      delete(handles.htext(i));
    end
  end
  
  handles.template = [];
  handles.npoints = 0;
  
end

handles = SetTemplate(handles);

handles.labeledpos = nan([handles.npoints,2,handles.nframes,handles.nanimals]);
handles.labeledpos(:,:,handles.f,handles.animal) = handles.template;
handles.islocked = false(handles.nframes,handles.nanimals);
handles.pointselected = false(1,handles.npoints);

delete(handles.posprev(ishandle(handles.posprev)));
handles.posprev = nan(1,handles.npoints);
for i = 1:handles.npoints,
  handles.posprev(i) = plot(handles.axes_prev,nan,nan,'+','Color',handles.templatecolors(i,:),'MarkerSize',8);%,...
    %'KeyPressFcn',handles.keypressfcn);
end

guidata(hObject,handles);

function CloseImContrast(hObject)

handles = guidata(hObject);
clim = get(handles.axes_curr,'CLim');
handles.minv = clim(1);
handles.maxv = clim(2);
set(handles.axes_prev,'CLim',[handles.minv,handles.maxv]);
set(handles.axes_curr,'CLim',[handles.minv,handles.maxv]);
delete(handles.adjustbrightness_listener);
guidata(hObject,handles);


% --------------------------------------------------------------------
function menu_setup_adjustbrightness_Callback(hObject, eventdata, handles)
% hObject    handle to menu_setup_adjustbrightness (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%set(handles.axes_curr,'CLim',[min(handles.imcurr(:)),max(handles.imcurr(:))]);
hcontrast = imcontrast_kb(handles.axes_curr);
handles.adjustbrightness_listener = addlistener(hcontrast,'ObjectBeingDestroyed',@(x,y) CloseImContrast(hObject));
guidata(hObject,handles);



% --- Executes when user attempts to close figure.
function figure_CloseRequestFcn(hObject, eventdata, handles)
% hObject    handle to figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: delete(hObject) closes the figure
CloseGUI(handles);


function CloseGUI(handles)

res = questdlg('Save before closing?');
if strcmpi(res,'Cancel'),
  return;
elseif strcmpi(res,'Yes'),
  SaveState(handles);
end

delete(handles.figure);

function handles = LoadLabels(handles,filename)

global LARVALABELERSAVEFILE;

if nargin < 2 || isempty(filename),
  
  if isempty(LARVALABELERSAVEFILE),
    defaultfile = '';
  else
    defaultfile = LARVALABELERSAVEFILE;
  end
  [f,p] = uigetfile('*.mat','Load file...',defaultfile);
  if ~ischar(f),
    return;
  end
  filename = fullfile(p,f);
end

if ~exist(filename,'file'),
  warndlg(sprintf('File %s does not exist',filename),'File does not exist','modal');
  return;
end

LARVALABELERSAVEFILE = filename;

handles.savefile = filename;
savedata = load(handles.savefile);
fns = fieldnames(savedata);
olddata = struct;
for i = 1:numel(fns),
  fn = fns{i};
  olddata.(fn) = handles.(fn);
  handles.(fn) = savedata.(fn);
end

if isfield(handles,'fid') && isnumeric(handles.fid) && handles.fid > 0,
  fclose(handles.fid);
end
handles = InitializeVideo(handles);
handles.npoints = size(handles.template,1);
handles.f_im = nan;
handles.fprev_im = nan;
handles.templatecolors = jet(handles.npoints)*.5+.5;

handles = UpdateFrame(handles);

% --------------------------------------------------------------------
function menu_file_load_Callback(hObject, eventdata, handles)
% hObject    handle to menu_file_load (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

res = questdlg('Save before closing current video?');
if strcmpi(res,'Cancel'),
  return;
elseif strcmpi(res,'Yes'),
  handles = SaveState(handles);
end

handles = LoadLabels(handles);

guidata(hObject,handles);

function moviefile = SelectVideo()
% loadMovie

% --- Executes on key press with focus on figure and none of its controls.
function figure_KeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure (see GCBO)
% eventdata  structure with the following fields (see FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)

%fprintf('Pressed %s, > %s<\n',eventdata.Key,sprintf('%s ',eventdata.Modifier{:}));
switch eventdata.Key,
  case 'rightarrow',
    if any(handles.pointselected),
      xlim = get(handles.axes_curr,'XLim');
      dx = diff(xlim);
      if ismember('shift',eventdata.Modifier),
        dx = dx / 50;
      else
        dx = dx / 500;
      end
      for i = find(handles.pointselected),
        x = get(handles.hpoly(i),'XData');
        handles.labeledpos(i,1,handles.f,handles.animal) = x + dx;
        set(handles.hpoly(i),'XData',handles.labeledpos(i,1,handles.f,handles.animal));
        tpos = get(handles.htext(i),'Position');
        tpos(1) = handles.labeledpos(i,1,handles.f,handles.animal)+handles.dt2p;
        set(handles.htext(i),'Position',tpos);
        guidata(hObject,handles);
      end
    else
      if ismember('control',eventdata.Modifier),
        df = 10;
      else
        df = 1;
      end
      f = min(handles.f+df,handles.nframes);
      if f ~= handles.f,
        handles.f = f;
        handles = UpdateFrame(handles,hObject);
        guidata(hObject,handles);
      end
    end
  case 'equal',
    if ismember('control',eventdata.Modifier),
      df = 10;
    else
      df = 1;
    end
    f = min(handles.f+df,handles.nframes);
    if f ~= handles.f,
      handles.f = f;
      handles = UpdateFrame(handles,hObject);
      guidata(hObject,handles);
    end
  case 'hyphen',
    if ismember('control',eventdata.Modifier),
      df = 10;
    else
      df = 1;
    end
    f = max(handles.f-df,1);
    if f ~= handles.f,
      handles.f = f;
      handles = UpdateFrame(handles,hObject);
      guidata(hObject,handles);
    end
  case 'leftarrow',
    if any(handles.pointselected),
      xlim = get(handles.axes_curr,'XLim');
      dx = diff(xlim);
      if ismember('shift',eventdata.Modifier),
        dx = dx / 50;
      else
        dx = dx / 500;
      end
      for i = find(handles.pointselected),
        x = get(handles.hpoly(i),'XData');
        handles.labeledpos(i,1,handles.f,handles.animal) = x - dx;
        set(handles.hpoly(i),'XData',handles.labeledpos(i,1,handles.f,handles.animal));
        tpos = get(handles.htext(i),'Position');
        tpos(1) = handles.labeledpos(i,1,handles.f,handles.animal)+handles.dt2p;
        set(handles.htext(i),'Position',tpos);

        guidata(hObject,handles);
      end
    else
      if ismember('control',eventdata.Modifier),
        df = 10;
      else
        df = 1;
      end
      f = max(handles.f-df,1);
      if f ~= handles.f,
        handles.f = f;
        handles = UpdateFrame(handles,hObject);
        guidata(hObject,handles);
      end
    end
  case 'uparrow',
    if any(handles.pointselected),
      ylim = get(handles.axes_curr,'YLim');
      dy = diff(ylim);
      if ismember('shift',eventdata.Modifier),
        dy = dy / 50;
      else
        dy = dy / 500;
      end
      for i = find(handles.pointselected),
        y = get(handles.hpoly(i),'YData');
        handles.labeledpos(i,2,handles.f,handles.animal) = y - dy;
        set(handles.hpoly(i),'YData',handles.labeledpos(i,2,handles.f,handles.animal));
        tpos = get(handles.htext(i),'Position');
        tpos(2) = handles.labeledpos(i,2,handles.f,handles.animal);
        set(handles.htext(i),'Position',tpos);

        guidata(hObject,handles);
      end
    end
  case 'downarrow',
    if any(handles.pointselected),
      ylim = get(handles.axes_curr,'YLim');
      dy = diff(ylim);
      if ismember('shift',eventdata.Modifier),
        dy = dy / 50;
      else
        dy = dy / 500;
      end
      for i = find(handles.pointselected),
        y = get(handles.hpoly(i),'YData');
        handles.labeledpos(i,2,handles.f,handles.animal) = y + dy;
        set(handles.hpoly(i),'YData',handles.labeledpos(i,2,handles.f,handles.animal));
        tpos = get(handles.htext(i),'Position');
        tpos(2) = handles.labeledpos(i,2,handles.f,handles.animal);
        set(handles.htext(i),'Position',tpos);

        guidata(hObject,handles);
      end
    end
  case 'l'
    set(handles.tbAccept,'Value',~get(handles.tbAccept,'Value'));
    togglebutton_lock_Callback(handles.tbAccept,[],handles);
  case 'r'
    pushbutton_template_Callback(hObject,eventdata,handles);
  case '1'
    if numel(handles.pointselected)<1, return; end
    inval = handles.pointselected(1);
    handles.pointselected(:) = false;
    handles.pointselected(1) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '2'
    if numel(handles.pointselected)<2, return; end
    inval = handles.pointselected(2);
    handles.pointselected(:) = false;
    handles.pointselected(2) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '3'
    if numel(handles.pointselected)<3, return; end
    inval = handles.pointselected(3);
    handles.pointselected(:) = false;
    handles.pointselected(3) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '4'
    if numel(handles.pointselected)<4, return; end
    inval = handles.pointselected(4);
    handles.pointselected(:) = false;
    handles.pointselected(4) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '5'
    if numel(handles.pointselected)<5, return; end
    inval = handles.pointselected(5);
    handles.pointselected(:) = false;
    handles.pointselected(5) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '6'
    if numel(handles.pointselected)<6, return; end
    inval = handles.pointselected(6);
    handles.pointselected(:) = false;
    handles.pointselected(6) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '7'
    if numel(handles.pointselected)<7, return; end
    inval = handles.pointselected(7);
    handles.pointselected(:) = false;
    handles.pointselected(7) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '8'
    if numel(handles.pointselected)<8, return; end
    inval = handles.pointselected(8);
    handles.pointselected(:) = false;
    handles.pointselected(8) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '9'
    if numel(handles.pointselected)<9, return; end
    inval = handles.pointselected(9);
    handles.pointselected(:) = false;
    handles.pointselected(9) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);
  case '0'
    if numel(handles.pointselected)<10, return; end
    inval = handles.pointselected(10);
    handles.pointselected(:) = false;
    handles.pointselected(10) = ~inval;
    UpdatePointSelected(handles);    
    guidata(hObject,handles);

end


% --- Executes on mouse motion over figure - except title and menu.
function figure_WindowButtonMotionFcn(hObject, eventdata, handles)
% hObject    handle to figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

disp('wbmf');

if ~isfield(handles,'motionobj') || isempty(handles.motionobj), return; end

if isnumeric(handles.motionobj),
  handles.didmovepoint = true;
  tmp = get(handles.axes_curr,'CurrentPoint');
  pos = tmp(1,1:2);
  set(handles.hpoly(handles.motionobj),'XData',pos(1),'YData',pos(2));
  pos(1) = pos(1) + handles.dt2p;
  set(handles.htext(handles.motionobj),'Position',pos);
  UpdateLabels(pos,hObject,handles.motionobj,handles);
end


% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure_WindowButtonUpFcn(hObject, eventdata, handles)
% hObject    handle to figure (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

disp('wbup');

if ~isempty(handles.motionobj) && isnumeric(handles.motionobj),
  if ~handles.didmovepoint,
    handles.pointselected(handles.motionobj) = ~handles.pointselected(handles.motionobj);
    UpdatePointSelected(handles,handles.motionobj);    
  end
  handles.motionobj = [];
  guidata(hObject,handles);
end

if any(handles.pointselected)
  assert(numel(find(handles.pointselected))==1,'More than one point cannot be selected');
  tmp = get(handles.axes_curr,'CurrentPoint');
  pos = tmp(1,1:2);
  set(handles.hpoly(handles.pointselected),'XData',pos(1),'YData',pos(2));
  set(handles.htext(handles.pointselected),'Position',[pos(1)+handles.dt2p pos(2)]);
  UpdateLabels(pos,hObject,find(handles.pointselected),handles);
  handles.pointselected(:) = false;
  guidata(hObject,handles)
  UpdatePointSelected(handles);
end

function UpdatePointSelected(handles,i)
if nargin<2,
  i = 1:numel(handles.pointselected);
end

for ndx = 1:numel(i)
  if handles.pointselected(i(ndx)),
    set(handles.hpoly(i(ndx)),'LineWidth',3,'Marker','x');
%     set(handles.htext(i(ndx)),'LineWidth',3,'Visible','off');
  else
    set(handles.hpoly(i(ndx)),'LineWidth',3,'Marker','+','Visible','on');
    set(handles.htext(i(ndx)),'Visible','on');
  end
end

% --------------------------------------------------------------------
function menu_help_Callback(hObject, eventdata, handles)
% hObject    handle to menu_help (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --------------------------------------------------------------------
function menu_help_keyboardshortcuts_Callback(hObject, eventdata, handles)
% hObject    handle to menu_help_keyboardshortcuts (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

s = {};
s{end+1} = '* When template point is selected, LEFT, RIGHT, UP, and DOWN move the selected point a small amount.';
s{end+1} = '* When template point is selected, CTRL+LEFT, RIGHT, UP, and DOWN move the selected point a large amount.';
s{end+1} = '* When no template point is selected, LEFT and RIGHT decrement and increment the frame shown.';
s{end+1} = '* MINUS (-) and EQUAL (=) always decrement and increment the frame shown.';
s{end+1} = '* When no template point is selected, CTRL+LEFT and CTRL+RIGHT decrease and increase the frame shown by 10.';
s{end+1} = '* CTRL+MINUS and CTRL+EQUAL decrease and increase the frame shown by 10.';
s{end+1} = '* L toggles the lock state for the current frame.';

msgbox(s,'Keyboard shortcuts','help','modal');

% --- Executes on button press in pushbutton_template.
function pushbutton_template_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton_template (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

trxndx = handles.f - handles.trx(handles.animal).firstframe+1;
curlocx = handles.trx(handles.animal).x(trxndx);
curlocy = handles.trx(handles.animal).y(trxndx);
dtheta = handles.trx(handles.animal).theta(trxndx)-handles.templatetheta;
rotmat = [cos(dtheta) -sin(dtheta); sin(dtheta) cos(dtheta)];
for i =1 :numel(handles.hpoly)
  dx = handles.template(i,1) - handles.templateloc(1);
  dy = handles.template(i,2) - handles.templateloc(2);
  rot = rotmat*[dx;dy];
  handles.labeledpos(i,1,handles.f,handles.animal) = curlocx + rot(1);
  handles.labeledpos(i,2,handles.f,handles.animal) = curlocy + rot(2);
  set(handles.hpoly(i),'XData',handles.labeledpos(i,1,handles.f,handles.animal),...
    'YData',handles.labeledpos(i,2,handles.f,handles.animal));
  tpos = [handles.labeledpos(i,1,handles.f,handles.animal)+handles.dt2p;...
          handles.labeledpos(i,2,handles.f,handles.animal)];
  set(handles.htext(i),'Position',tpos);

end
guidata(hObject,handles);


% --------------------------------------------------------------------
function import_template_Callback(hObject, eventdata, handles)
% hObject    handle to import_template (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global LARVALABELERSAVEFILE;

if isempty(LARVALABELERSAVEFILE),
  defaultfile = '';
else
  defaultfile = LARVALABELERSAVEFILE;
end

[f,p] = uigetfile('*.mat','Import template from...',defaultfile);
if ~ischar(f),
  return;
end
filename = fullfile(p,f);

if ~exist(filename,'file'),
  warndlg(sprintf('File %s does not exist',filename),'File does not exist','modal');
  return;
end

L = load(filename);
assert(isfield(L,'template'),'The file does not have any template');

handles.template = L.template;
handles.npoints = size(handles.template,1);
handles.templatecolors = jet(handles.npoints);%*.5+.5;

for ndx = 1:size(L.template,1)
  x = L.template(ndx,1);
  y = L.template(ndx,2);
  handles.hpoly(ndx) = plot(handles.axes_curr,x,y,'w+','MarkerSize',20,'LineWidth',3);
  handles.htext(ndx) = text(x+handles.dt2p,y,num2str(numel(handles.hpoly)),'Parent',handles.axes_curr);%,...
  set(handles.hpoly(ndx),'Color',handles.templatecolors(ndx,:),...
    'ButtonDownFcn',@(hObject,eventdata) PointButtonDownCallback(hObject,eventdata,handles.figure,ndx));
  %addNewPositionCallback(handles.hpoly(i),@(pos) UpdateLabels(pos,handles.figure,i));
  set(handles.htext(ndx),'Color',handles.templatecolors(ndx,:));

end
 
handles.labeledpos = nan([handles.npoints,2,handles.nframes,handles.nanimals]);
handles.labeledpos(:,:,handles.f,handles.animal) = handles.template;
handles.islocked = false(handles.nframes,handles.nanimals);
handles.pointselected = false(1,handles.npoints);

delete(handles.posprev(ishandle(handles.posprev)));
handles.posprev = nan(1,handles.npoints);
for i = 1:handles.npoints,
  handles.posprev(i) = plot(handles.axes_prev,nan,nan,'+','Color',handles.templatecolors(i,:),'MarkerSize',8);%,...
    %'KeyPressFcn',handles.keypressfcn);
end

if isfield(L,'templateloc')
  handles.templateloc = L.templateloc;
  handles.templatetheta = L.templatetheta;
  guidata(hObject,handles);
  pushbutton_template_Callback(hObject,eventdata,handles);
else
  guidata(hObject,handles);
end
