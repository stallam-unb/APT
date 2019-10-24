classdef MFTSetEnum < MFTSet
  % Some "canned" MFTSets for eg track pulldown menu.
  
  enumeration
    CurrMovTgtEveryFrame (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.AllFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.CurrTgt)
    CurrMovTgtEveryLblFrame (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.LabeledFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.CurrTgt)
    CurrMovTgtEveryNFramesSmall (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.AllFrm,...
      FrameDecimationVariable.EveryNFrameSmall,TargetSetVariable.CurrTgt)
    CurrMovTgtEveryNFramesLarge (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.AllFrm,...
      FrameDecimationVariable.EveryNFrameLarge,TargetSetVariable.CurrTgt)
    CurrMovTgtSelectedFrames (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.SelFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.CurrTgt)
    CurrMovTgtSelectedFramesEveryNFramesSmall (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.SelFrm,...
      FrameDecimationVariable.EveryNFrameSmall,TargetSetVariable.CurrTgt)
    CurrMovTgtSelectedFramesEveryNFramesLarge (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.SelFrm,...
      FrameDecimationVariable.EveryNFrameLarge,TargetSetVariable.CurrTgt)
    CurrMovTgtNearCurrFrame (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.WithinCurrFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.CurrTgt)
    CurrMovTgtNearCurrFrameEveryNFramesSmall (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.WithinCurrFrm,...
      FrameDecimationVariable.EveryNFrameSmall,TargetSetVariable.CurrTgt)
    CurrMovTgtNearCurrFrameEveryNFramesLarge (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.WithinCurrFrm,...
      FrameDecimationVariable.EveryNFrameLarge,TargetSetVariable.CurrTgt)
    
    CurrMovAllTgtsEveryFrame (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.AllFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts)
    CurrMovAllTgtsEveryLblFrame (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.LabeledFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts)
    CurrMovAllTgtsEveryNFramesSmall (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.AllFrm,...
      FrameDecimationVariable.EveryNFrameSmall,TargetSetVariable.AllTgts)
    CurrMovAllTgtsEveryNFramesLarge (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.AllFrm,...
      FrameDecimationVariable.EveryNFrameLarge,TargetSetVariable.AllTgts)
    CurrMovAllTgtsSelectedFrames (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.SelFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts)
    CurrMovAllTgtsSelectedFramesEveryNFramesSmall (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.SelFrm,...
      FrameDecimationVariable.EveryNFrameSmall,TargetSetVariable.AllTgts)
    CurrMovAllTgtsSelectedFramesEveryNFramesLarge (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.SelFrm,...
      FrameDecimationVariable.EveryNFrameLarge,TargetSetVariable.AllTgts)
    CurrMovAllTgtsNearCurrFrame (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.WithinCurrFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts)
    CurrMovAllTgtsNearCurrFrameEveryNFramesSmall (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.WithinCurrFrm,...
      FrameDecimationVariable.EveryNFrameSmall,TargetSetVariable.AllTgts)
    CurrMovAllTgtsNearCurrFrameEveryNFramesLarge (...
      MovieIndexSetVariable.CurrMov,FrameSetVariable.WithinCurrFrm,...
      FrameDecimationVariable.EveryNFrameLarge,TargetSetVariable.AllTgts)
    
    AllMovAllLabeled (...
      MovieIndexSetVariable.AllMov,FrameSetVariable.LabeledFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
    AllMovAllLabeled2 (...
      MovieIndexSetVariable.AllMov,FrameSetVariable.Labeled2Frm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
    AllMovAllTgtAllFrm (...
      MovieIndexSetVariable.AllMov,FrameSetVariable.AllFrm,...
      FrameDecimationFixed.EveryFrame,TargetSetVariable.AllTgts);
  end
  
  properties (Constant)
    TrackingMenuNoTrx = [...
      MFTSetEnum.CurrMovTgtEveryLblFrame;...
      MFTSetEnum.CurrMovTgtEveryFrame;...
      ... % MFTSetEnum.CurrMovTgtEveryNFramesLarge;...
      MFTSetEnum.CurrMovTgtSelectedFrames;...
      ... % MFTSetEnum.CurrMovTgtSelectedFramesEveryNFramesLarge;...
      MFTSetEnum.CurrMovTgtNearCurrFrame]; % ...
      % MFTSetEnum.CurrMovTgtNearCurrFrameEveryNFramesLarge];      
    TrackingMenuTrx = [...
      MFTSetEnum.CurrMovTgtEveryLblFrame;...
      MFTSetEnum.CurrMovTgtEveryFrame;...
      ... % MFTSetEnum.CurrMovTgtEveryNFramesLarge;...
      MFTSetEnum.CurrMovTgtSelectedFrames;...
      ... % MFTSetEnum.CurrMovTgtSelectedFramesEveryNFramesLarge;...
      MFTSetEnum.CurrMovTgtNearCurrFrame;...
      ... % MFTSetEnum.CurrMovTgtNearCurrFrameEveryNFramesLarge;...
      MFTSetEnum.CurrMovAllTgtsEveryLblFrame;...
      MFTSetEnum.CurrMovAllTgtsEveryFrame;...
      ... % MFTSetEnum.CurrMovAllTgtsEveryNFramesLarge;...
      MFTSetEnum.CurrMovAllTgtsSelectedFrames;...
      ... % MFTSetEnum.CurrMovAllTgtsSelectedFramesEveryNFramesLarge;...
      MFTSetEnum.CurrMovAllTgtsNearCurrFrame;]; % ...
      %MFTSetEnum.CurrMovAllTgtsNearCurrFrameEveryNFramesLarge];
  end
    
end