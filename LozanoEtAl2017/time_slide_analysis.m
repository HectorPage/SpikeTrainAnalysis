function ATI = time_slide_analysis(HeadDirection,original_SpikeTimes, max_time ,AngHeadVel)

shifts = 0:0.02:0.16; %in seconds, so these are 20ms jumps
R_PFD = zeros(numel(shifts),1);
L_PFD = zeros(numel(shifts),1);

HD_binedges = 0:6:360;
num_HD_bins = numel(HD_binedges)-1;

smoothing_kernel = 5;

for idx = 1:numel(shifts)
    
    %% shift SpikeTimes by the time shift
    SpikeTimes = original_SpikeTimes + shifts(idx);
    
    SpikeTimes(SpikeTimes<0) = SpikeTimes(SpikeTimes<0) + max_time;
    SpikeTimes(SpikeTimes>max_time) = SpikeTimes(SpikeTimes>max_time) - max_time;
    
    %% work out tuning curves
    left_AHVbyHDSpikes = zeros(1, num_HD_bins); %array for the spikecounts
    left_AHVbyHDDwell = zeros(1, num_HD_bins); %array for dwelltime
    
    for HDindex = 1:num_HD_bins
        
        %find time bins where both variables are within bin range
        conjunctive_timebins = find(HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1) & AngHeadVel<0);
        
        left_AHVbyHDDwell(HDindex) = left_AHVbyHDDwell(HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
        
        for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
            lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
            upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
            
            left_AHVbyHDSpikes(HDindex) = left_AHVbyHDSpikes(HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
        end
        
    end
    
    left_AHVbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,left_AHVbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
    left_AHVbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,left_AHVbyHDSpikes);
    
    left_AHVbyHDRates = left_AHVbyHDSpikes./left_AHVbyHDDwell;
    
    
    right_AHVbyHDSpikes = zeros(1, num_HD_bins); %array for the spikecounts
    right_AHVbyHDDwell = zeros(1, num_HD_bins); %array for dwelltime
    
    for HDindex = 1:num_HD_bins
        
        %find time bins where both variables are within bin range
        conjunctive_timebins = find(HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1) & AngHeadVel>0);
        
        right_AHVbyHDDwell(HDindex) = right_AHVbyHDDwell(HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
        
        for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
            lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
            upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
            
            right_AHVbyHDSpikes(HDindex) = right_AHVbyHDSpikes(HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
        end
        
    end
    
    right_AHVbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,right_AHVbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
    right_AHVbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,right_AHVbyHDSpikes);
    
    right_AHVbyHDRates = right_AHVbyHDSpikes./right_AHVbyHDDwell;
    
    %% get PFDs of the two tuning curves
    binmeans_sin = sind(3:6:357)'; %sine
    binmeans_cos = cosd(3:6:357)'; %cosine
      
    vector_1_R = sum(bsxfun(@times,right_AHVbyHDRates',binmeans_sin),1);
    vector_2_R = sum(bsxfun(@times,right_AHVbyHDRates',binmeans_cos),1);
    %Get atan2 (four-quadrant corrected) angles
    RPV = atan2d(vector_1_R,vector_2_R);
    %Get in the right range (0-360)
    RPV(RPV<0) = RPV(RPV<0)  + 360;
    
    R_PFD(idx) = RPV;
    
    vector_1_L = sum(bsxfun(@times,left_AHVbyHDRates',binmeans_sin),1);
    vector_2_L = sum(bsxfun(@times,left_AHVbyHDRates',binmeans_cos),1);
    %Get atan2 (four-quadrant corrected) angles
    LPV = atan2d(vector_1_L,vector_2_L);
    %Get in the right range (0-360)
    LPV(LPV<0) = LPV(LPV<0)  + 360;
    
    L_PFD(idx) = LPV;
    
    
    
end

%% Now get difference in PFD
PFD_diffs = atan2d(sind(L_PFD-R_PFD),cosd(L_PFD-R_PFD));

% %% Now plot PFD difference vs time shifts
% figure()
% plot(shifts,PFD_diffs);

%% Now need to find point where PFD_diffs = 0
%Fit a line to PFD diffs

line = polyfit(PFD_diffs',shifts,1); %just a straight line for simplicity
ATI = polyval(line,0); %get time shift when PFD_diff = 0, based on fitted line


end


