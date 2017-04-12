function yave_analysisV2(mode)
%close all;

%% SETTING UP DATA

startpath = pwd; %come back here afterwards
%Add all the necessary functions to path
%/Volumes/ibn/Jeffery lab/Hector Page is my filestore folder

addpath(genpath('/Volumes/ibn/Jeffery lab/Hector Page/flip cell work/RoddyCodes'));

%now read in unique sessions
if strcmpi(mode,'first')
    session_file = '/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/new_sessions_to_analyse.xlsx';
elseif strcmpi(mode,'second')
    session_file = '/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/new_sessions_to_analyse2.xlsx';
else
    session_file = '/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/new_sessions_to_analyse3.xlsx';
end
%
if strcmpi(mode,'debug')
    session_file = '/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/problem_session.xlsx';
end

[~,session_text,~] = xlsread(session_file);
session_text = session_text(~cellfun('isempty',session_text)); %excel file contains blank lines, this drops those elements
total_sessions = numel(session_text);

analysis_targets = zeros(total_sessions,3); %this holds targets of analysis in format: Rat Tetrode Cluster

%string extraction
areas = cellfun(@(x) x(1:3),session_text,'un',0); %extract area from cell array
cards = cellfun(@(x) x(10:12),session_text,'un',0); %extract card type from cell array
dates = cellfun(@(x) x(14:21),session_text,'un',0); %extract date type from cell array


for target = 1:total_sessions
    %format of the string is 'POS\r638\B-B\06.07.15\,4,1'
    [analysis_targets(target,1:3),~] = sscanf(session_text{target},[areas{target},'\\r%f\\',cards{target},...
        '\\',dates{target},'\\,%f,%f']);
end

%number extraction
rats = analysis_targets(:,1);
tetrodes = analysis_targets(:,2);
clusters = analysis_targets(:,3);

clear analysis_targets;

%now read in the trials for each session
%trial_file = '/Volumes/ibn/Jeffery lab/Hector Page/Yave data/trials_within_sessions.xlsx';

%below is the updated version from Kate
trial_file = '/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/new_trials_to_analyse.xlsx';

[~,trial_text,~] = xlsread(trial_file);
trial_text = trial_text(~cellfun('isempty',trial_text)); %excel file contains blank lines, this drops those elements

%% Split data by POS and RSC
trial_areas = cellfun(@(x) x(1:3),trial_text,'un',0); %extract area from cell array
POS_trials = numel(find(ismember(trial_areas,'POS')));
RSC_trials = numel(trial_areas) - POS_trials;

%Create cell array to hold names of RSC and POS trials
% RSC_trial_names = cell(RSC_trials,1);
% POS_trial_names = cell(POS_trials,1);
%
% RSC_ISI_half_peaks = zeros(RSC_trials,1);
% POS_ISI_half_peaks = zeros(POS_trials,1);
%
% RSC_ISI_peaks = zeros(RSC_trials,1);
% POS_ISI_peaks = zeros(POS_trials,1);
%
% RSC_ISI_constant_a = zeros(RSC_trials,1);
% POS_ISI_constant_a = zeros(POS_trials,1);
%
% RSC_ISI_constant_b = zeros(RSC_trials,1);
% POS_ISI_constant_b = zeros(POS_trials,1);
%
% RSC_count = 0;
% POS_count = 0;
%
% RSC_ISIs = 0;
% POS_ISIs = 0;

%% Session Loop

%% Recording stats collapsed across sessions
running_speed_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
shuffled_running_speed_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
running_speed_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2
shuffled_running_speed_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2

left_turn_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
shuffled_left_turn_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
left_turn_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2
shuffled_left_turn_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2

right_turn_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
shuffled_right_turn_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
right_turn_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2
shuffled_right_turn_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2

%The fake stats
fake_left_turn_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
fake_left_turn_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2
fake_right_turn_stats_all_sessions = zeros(total_sessions,4); %slope intercept R^2 r
fake_right_turn_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2

fake_running_speed_stats_all_sessions = zeros(total_sessions,4);  %slope intercept R^2 r
fake_running_speed_exp_stats_all_sessions = zeros(total_sessions,3); %a b R^2

avg_max_rate = zeros(total_sessions,1); %mean of max rate across all trials

for session = 1:total_sessions %loop over all the sessions
    
    %first half of wildcard for session
    session_wildcard1 = [areas{session},'\r',num2str(rats(session)),'\',cards{session},'\',dates{session},...
        '\r',num2str(rats(session)),'_',dates{session}];
    
    %now add on correct tetrode and cluster
    session_wildcard2 = [num2str(tetrodes(session)),',',num2str(clusters(session))]; %now search all the remainder for the right tetrode and cluster
    
    %search for things starting with wildcard1 and ending with wildcard 2
    x = regexp(trial_text,regexptranslate('wildcard',[session_wildcard1,'*,',session_wildcard2])); %creates cell array of 1 for trials in this session or [otherwise]
    x = find(~cellfun(@isempty,x)); %gets index of non-empty bits of cell array
    
    this_session_trials = trial_text(x);
    
    %get the names/numbers of the trials in this session
    trial_labels = cellfun(@(x) x(end-5:end-4),this_session_trials,'un',0); %extract trials from cell array
    
    %need to check for double-digit trial labels
    for index = 1:numel(trial_labels)
        if(~strncmpi(trial_labels{index},'b',1) && ~strncmpi(trial_labels{index},'t',1)) %if no b or t at start of filename
            trial_labels{index} = ['t',trial_labels{index}]; %append the t prefix
            filepath = ['/Volumes/ibn/Jeffery lab/Hector Page/Yave data/Raw Dacq data/'...
                ,areas{session},'/r',num2str(rats(session)),...
                '/',cards{session},'/',dates{session}]; %area rat card date
            cd(filepath);
            
            flnmchk = ['r',num2str(rats(session)),'_',dates{session},'_',trial_labels{index},'.set'];
            if exist(flnmchk,'file') ~= 2 %if that trial file doesn't exist
                trial_labels{index}(1) = 'b';
            end
        end
    end
    
    num_trials_this_session = numel(this_session_trials);
    
    
    cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis');
    savepath = ['/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/',session_text{session}];
    
    if ~exist(session_text{session}, 'dir')
        fprintf('\t...making Session Folder\n');
        mkdir(session_text{session});
    end
    
    cd(savepath);
    
    %record order that trials go in data summary structures for this
    %session
    fileID = fopen('trial_data_ordering.txt','w');
    formatSpec = '%s\n';
    for idx = 1:num_trials_this_session
        fprintf(fileID,formatSpec,trial_labels{idx,:});
    end
    fclose(fileID);
    
    max_rate  = zeros(num_trials_this_session,1);
    
    %     %% for LFP analysis
    %     LFP_frequency = zeros(num_trials_this_session,2); %frequency, and then power
    %     speed_theta_stats = zeros(num_trials_this_session,4); %in format, slope-intercept-r-p
    %     theta_ratio = zeros(num_trials_this_session,1);
    %     alt_theta_ratio = zeros(num_trials_this_session,1);
    %     intrinsic_freq_stats = zeros(num_trials_this_session,4);
    %
    %     session_ISI_peaks = zeros(num_trials_this_session,2); %in format, half peak time onwards from peak, peak time
    %     session_ISI_exp_params = zeros(num_trials_this_session,2); %in format coeff1 coeff2
    %
    %     %% for ISIs
    %     this_session_ISIs = 0;
    %
    %     ISI_peak = 0;
    %     ISI_half_peak = 0;
    %     ISI_constant_a = 0;
    %     ISI_constant_b = 0;
    %
    %     %Data strutures to count number of ISI observations per trial
    %     ISI_spikecount = 0;
    %
    %     %Holding histogram bin means
    %     bin_counts_per_trial= zeros(num_trials_this_session,500);
    
    filepath = ['/Volumes/ibn/Jeffery lab/Hector Page/Yave data/Raw Dacq data/',areas{session},'/r',num2str(rats(session)),...
        '/',cards{session},'/',dates{session}]; %area rat card date
    cd(filepath);
    
    for trialnumber = 1:num_trials_this_session %do all the plotting for each trial of the data
        
        disp('------------------------------------');
        disp(['Progress = Trial:', num2str(((trialnumber-1)/num_trials_this_session)*100),'% Session:',num2str(session-1/total_sessions),'%']);
        disp('------------------------------------');
        
        filename = ['r',num2str(rats(session)),'_',dates{session},'_',trial_labels{trialnumber}];
        full_filename = strcat(filename,'.set');
        %% READING DATA
        
        disp('------------------------------------');
        fprintf('Fetching DACQ data...');
        mtint = readAllDACQdata(filepath,full_filename);
        disp('------------------------------------');
        
        %%%%%%%%%%%%%% Get some variables that will be important later
        tmax = mtint.header{5,2};																% get the session duration stored in header info
        tmax = str2num(tmax);																	% tmax now = the length of the session (s)
        
        position = mtint.pos.xy_pixels;																% get the position data for the whole session
        posx = position(:,1);                                                                       % extract just the x coordinates
        posy = position(:,2);                                                                       % extract just the y coordinates
        
        head_direct = (mtint.pos.dir)';																% get the head direction information for the whole session
        head_direct = head_direct(:,1);
        
        pixel_ratio = mtint.pos.header{26,2};															% get the pixel ratio (pixels per meter)
        pixel_ratio = str2num(pixel_ratio);															% pixel ratio now = the pixels per meter of the tracking camera
        
        pos_tb = mtint.pos.header{16,2};															% get the position data sampling rate (should be 50hz) or 0.05s
        pos_tb = str2num(pos_tb(1:2))/1000;
        
        Spikes = mtint.tetrode(tetrodes(session)).cut==clusters(session);                                         %logical index of which spikes on this tetrode are clusters(session)
        SpikeTimes= mtint.tetrode(tetrodes(session)).ts(Spikes);                                       %indexing in to extract spiketimes for this cell
        
        %position data
        clu_assign = mtint.tetrode(tetrodes(session)).cut;												% clu_assign is a vector of numbers, one for each spike, each number corresponds to a cluster
        pos_assign = mtint.tetrode(tetrodes(session)).pos_sample;                                       % pos_assign is a vector of numbers, one for each spike, each number corresponds to a position data point
        n_spikes = length(find(clu_assign==clusters(session)));
        
        frate = n_spikes ./ tmax;                                                               %average firing rate over trial
        
        pox = double(posx);
        poy = double(posy);
        
        spx = double(posx(pos_assign(clu_assign == clusters(session))));
        spy = double(posy(pos_assign(clu_assign == clusters(session))));
        
        hd = double(head_direct(pos_assign(clu_assign == clusters(session))));
        
        com_min_x = min(pox);
        com_min_y = min(-poy);
        pox = pox - com_min_x;
        poy = -poy - com_min_y;
        spx = spx - com_min_x;
        spy = -spy - com_min_y;
        
        
        
        %% Record Spike Counts for reference
        %         fileID = fopen('SpikeCount.txt','w');
        %         formatSpec = '%f\n';
        %         fprintf(fileID,formatSpec,num2str(numel(SpikeTimes)));
        %         fclose(fileID);
        
        %
        %% Get Head Direction
        
        %         Vars=daVarsStruct;
        %         in_polar1 = make_in_struct_for_rm (mtint,tetrodes(session),clusters(session),50,Vars.rm.smthKernPos,Vars.rm.binSizePosCm,Vars.rm.binSizeDir,0,Vars,'dir');
        %         if size(in_polar1.posDir,2) ~= 1
        %             in_polar1.posDir = in_polar1.posDir';
        %         end
        %         in = in_polar1;
        %
        %         PolarMap_cell1 = GetRateMap (in);
        %         circRm_cell1=PolarMap_cell1.map; %mean firing rate for each bin
        %         HD_bin_centers = 3:6:357; %center of each bin
        
        
        
        %%
        %         %%%%%%%%%%%%%%% Plot: HD polar tuning curve (to check)
        %         figure()
        %         p = mmpolar(0: pi/30 : 2*pi,[circRm_cell1' circRm_cell1(1)'],...
        %             'Grid','off','RTickVisible','off','RTickLabelVisible','off','TTickLabelVisible',...
        %             'off','TTickVisible','off', 'Color','TTickSign','+','Border','off','Axis','off');
        %
        %         hold on
        %         line([0 0],[-1 1],'color','k');
        %         line([-1 1],[0 0],'color','k');
        %         hold off
        %
        %         set(p(1),'LineWidth',2,'color','b');
        %
        %         cd(savepath);   %put into results folder
        %         saveas(gcf,[trial_labels{trialnumber},'_polar_plot'],'tif');
        %         close(gcf);
        
        %% Get  circular rate map, and get max rate from that
        Vars=daVarsStruct;
        in_polar1 = make_in_struct_for_rm (mtint,tetrodes(session),clusters(session),50,Vars.rm.smthKernPos,Vars.rm.binSizePosCm,Vars.rm.binSizeDir,0,Vars,'dir');
        if size(in_polar1.posDir,2) ~= 1
            in_polar1.posDir = in_polar1.posDir';
        end
        in = in_polar1;
        
        PolarMap_cell1 = GetRateMap (in);
        circRm_cell1=PolarMap_cell1.map; %mean firing rate for each bin
        HD_bin_centers = 3:6:357; %center of each bin
        
        max_rate(trialnumber) = max(circRm_cell1); %record of max rate to be averaged and used to normalise the slope data
        
        %% Work out directional firing range of the cell
        [~,sortIndex] = sort(circRm_cell1(:),'descend');
        
        peak1 = HD_bin_centers(sortIndex(1)); %this is taken as PFD of the cell - bit inaccurate but will do
        
        %Speed
        speed = mtint.pos.speed; %this is presumably speed in cm/s
        
        %running_speed_distributions(trialnumber,:) = [min(speed), max(speed), mean(speed), std(speed)]; %min max mean sd
        
        speed_binedges = [2:2:32];
        num_speed_bins = numel(speed_binedges)-1;
        
        peak1 = peak1 * (pi/180);
        
        %HD
        HeadDirection = in.posDir; %HD at 50ms sample rate
        HeadDirection = HeadDirection * (pi/180); %lol radians
        
        %Getting current HD relative to PFD - is it within 90 deg firing range
        %of the cell?
        within_firing_range = false(numel(HeadDirection),1);
        
        for index = 1:numel(HeadDirection)      %just for RSC HD cells
            
            distance1 = abs(atan2(sin(peak1-HeadDirection(index)),cos(peak1-HeadDirection(index)))); %absolute PFD change from trial 1 to trial 5
            distance2 = (2*pi) - distance1;
            
            if(distance1<distance2)
                distance = distance1;
            else
                distance = distance2;
            end
            
            if(distance<(pi/4))
                within_firing_range(index) = 1; %if within range, this variable is 1 (true) otherwise 0 (false)
            end
        end
        
        
        %% Running speed spike binning
        
        Running_Speed_Spikes = zeros(num_speed_bins, 1); %array for the spikecounts
        Running_Speed_Dwell = zeros(num_speed_bins, 1); %array for dwelltime
        
        for speedindex = 1:num_speed_bins
            
            %find time bins where both variables are within bin range
            conjunctive_timebins = find(speed>=speed_binedges(speedindex) & speed<speed_binedges(speedindex+1)...
                & within_firing_range); %changed from ~within_firing_range
            
            Running_Speed_Dwell(speedindex) = numel(conjunctive_timebins)/ 50;   %number of 20ms time bins, so /50 to get time in seconds
            
            for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                
                Running_Speed_Spikes(speedindex) = Running_Speed_Spikes(speedindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
            end
        end
        
        if trialnumber<2 %record the spikes and dwell per trial
            running_speed_spikes_all_trials = Running_Speed_Spikes;
            running_speed_dwell_all_trials = Running_Speed_Dwell;
        else
            running_speed_spikes_all_trials = running_speed_spikes_all_trials + Running_Speed_Spikes;
            running_speed_dwell_all_trials = running_speed_dwell_all_trials + Running_Speed_Dwell;
        end
        
        
        %Here is code to look at distribution of running speed over the
        %course of the trial
        %           cd(savepath);   %put into results folder
        %
        %
        %           [counts,~] = histcounts(speed,0:2:70);
        %           counts = counts./max(counts);
        %           bar(1:2:69,counts,'histc');
        %           ylabel('Norm Freq','Fontsize',24);
        %           xlabel('Running Speed (cm/s)','Fontsize',24);
        %           set(gca,'Fontsize',24);
        %           saveas(gcf,[trial_labels{trialnumber},'_running_speed_dist'],'jpg');
        
        
        
        %% Plot: Firing Rate vs. AHV spike binning (as above with running speed - only within PFD of cell
        AngHeadVel = zeros(numel(HeadDirection),1);
        HeadDirection = in.posDir; %previously converted to radians, this is the degrees
        
        for index = 1 : (numel(HeadDirection) - 1)
            distance = HeadDirection(index+1)-HeadDirection(index); %this works provided HeadDirection ranges 0-360
            
            %correcting distance to be signed shortest distance between two HDs
            if(distance<-180)
                distance = distance + (360);
            elseif(distance>180)
                distance = distance - (360);
            end
            
            AngHeadVel(index) = distance * 50; %multiplied by 50Hz to give deg/s
        end
        
        %Doing for R and L turns separately
        
        RAngHeadVel = AngHeadVel(AngHeadVel<500 & AngHeadVel>0);
        LAngHeadVel = AngHeadVel(AngHeadVel<0 & AngHeadVel>-500);
        
        %num_ahv_bins = 250; %0 to 500 in each direction, 2deg/s bins
        
        %changing the bins to see if it has an effect!
        
 
        
        [R_AHV_bincounts, R_AHV_binedges] = histcounts(RAngHeadVel,0:2:500); %bin counts is number of observations (20ms time intervals) in each bin
        [L_AHV_bincounts, L_AHV_binedges] = histcounts(LAngHeadVel,-500:2:0); %bin counts is number of observations (20ms time intervals) in each
        
        R_AHV_dwell = R_AHV_bincounts/50;  %time spent in AHV bin (divided by 50 as sample rate 50Hz) - could replace with pos_tb?
        L_AHV_dwell = L_AHV_bincounts/50;  %time spent in AHV bin (divided by 50 as sample rate 50Hz) - could replace with pos_tb?
        
        R_AHV_spikecount = zeros(1,numel(R_AHV_bincounts));
        L_AHV_spikecount = zeros(1,numel(L_AHV_bincounts));
        for index = 1: numel(R_AHV_bincounts) %same num in R as in L as histogram same proportions
            
            %get index of all time bins where AHV is within binrange, do for
            %non-corrected AHV to get the right index number to find spikes
            
            R_AHV_timebins = find(AngHeadVel>=R_AHV_binedges(index) & AngHeadVel<R_AHV_binedges(index+1) & within_firing_range);
            L_AHV_timebins = find(AngHeadVel>=L_AHV_binedges(index) & AngHeadVel<L_AHV_binedges(index+1) & within_firing_range);
            
            %now sum spikes that occur in those time bins
            
            for jdx = 1: numel(R_AHV_timebins)
                
                lower_timebin_R = R_AHV_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin_R = (R_AHV_timebins(jdx) + 1) * (20/1000);
                
                R_AHV_spikecount(index) = R_AHV_spikecount(index) + numel(SpikeTimes(SpikeTimes>=lower_timebin_R & SpikeTimes<upper_timebin_R));
                
            end
            
            for jdx = 1: numel(L_AHV_timebins)
                
                lower_timebin_L = L_AHV_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin_L = (L_AHV_timebins(jdx) + 1) * (20/1000);
                
                L_AHV_spikecount(index) = L_AHV_spikecount(index) + numel(SpikeTimes(SpikeTimes>=lower_timebin_L & SpikeTimes<upper_timebin_L));
                
            end
            
        end
        
        if trialnumber<2
            R_turn_spikes_all_trials = R_AHV_spikecount;
            R_turn_dwell_all_trials = R_AHV_dwell;
            
            L_turn_spikes_all_trials = L_AHV_spikecount;
            L_turn_dwell_all_trials = L_AHV_dwell;
        else
            R_turn_spikes_all_trials = R_turn_spikes_all_trials + R_AHV_spikecount;
            R_turn_dwell_all_trials = R_turn_dwell_all_trials + R_AHV_dwell;
            
            L_turn_spikes_all_trials = L_turn_spikes_all_trials + L_AHV_spikecount;
            L_turn_dwell_all_trials = L_turn_dwell_all_trials + L_AHV_dwell;
        end
        
        
        
        
        
        %% Doing a test for Kate - 10Hz regular spiking, to see if we have some inherent bias in the analysis, binning L and R turns
        %Create 10Hz spike train
        fake_SpikeTimes = 0.1:0.1:tmax;
        %Use PFD/within_firing_range filter of the actual data for this
        %trial - Do L and R analysis as if this was real SpikeTimes
        
        RAngHeadVel = AngHeadVel(AngHeadVel<500 & AngHeadVel>0);
        LAngHeadVel = AngHeadVel(AngHeadVel<0 & AngHeadVel>-500);
        
  
        [R_AHV_bincounts, R_AHV_binedges] = histcounts(RAngHeadVel,0:2:500); %bin counts is number of observations (20ms time intervals) in each bin
        [L_AHV_bincounts, L_AHV_binedges] = histcounts(LAngHeadVel,-500:2:0); %bin counts is number of observations (20ms time intervals) in each
        
        
        fake_R_AHV_spikecount = zeros(1,numel(R_AHV_bincounts));
        fake_L_AHV_spikecount = zeros(1,numel(L_AHV_bincounts));
        for index = 1: numel(R_AHV_bincounts) %same num in R as in L as histogram same proportions
            
            %get index of all time bins where AHV is within binrange, do for
            %non-corrected AHV to get the right index number to find spikes
            
            R_AHV_timebins = find(AngHeadVel>=R_AHV_binedges(index) & AngHeadVel<R_AHV_binedges(index+1) & within_firing_range);
            L_AHV_timebins = find(AngHeadVel>=L_AHV_binedges(index) & AngHeadVel<L_AHV_binedges(index+1) & within_firing_range);
            
            %now sum spikes that occur in those time bins
            
            for jdx = 1: numel(R_AHV_timebins)
                
                lower_timebin_R = R_AHV_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin_R = (R_AHV_timebins(jdx) + 1) * (20/1000);
                
                fake_R_AHV_spikecount(index) = fake_R_AHV_spikecount(index) + numel(fake_SpikeTimes(fake_SpikeTimes>=lower_timebin_R & fake_SpikeTimes<upper_timebin_R));
                
            end
            
            for jdx = 1: numel(L_AHV_timebins)
                
                lower_timebin_L = L_AHV_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin_L = (L_AHV_timebins(jdx) + 1) * (20/1000);
                
                fake_L_AHV_spikecount(index) = fake_L_AHV_spikecount(index) + numel(fake_SpikeTimes(fake_SpikeTimes>=lower_timebin_L & fake_SpikeTimes<upper_timebin_L));
                
            end
            
        end
        
        if trialnumber<2
            fake_R_turn_spikes_all_trials = fake_R_AHV_spikecount;
            fake_L_turn_spikes_all_trials = fake_L_AHV_spikecount;
        else
            fake_R_turn_spikes_all_trials = fake_R_turn_spikes_all_trials + fake_R_AHV_spikecount;
            fake_L_turn_spikes_all_trials = fake_L_turn_spikes_all_trials + fake_L_AHV_spikecount;
        end
        
        %% Using the fake 10Hz train to also look at running speed
        
          fake_Running_Speed_Spikes = zeros(num_speed_bins, 1); %array for the spikecounts
       
        
        for speedindex = 1:num_speed_bins
            
            %find time bins where both variables are within bin range
            conjunctive_timebins = find(speed>=speed_binedges(speedindex) & speed<speed_binedges(speedindex+1)...
                & within_firing_range); %changed from ~within_firing_range
 
            for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                
                fake_Running_Speed_Spikes(speedindex) = fake_Running_Speed_Spikes(speedindex) + numel(fake_SpikeTimes(fake_SpikeTimes>=lower_timebin & fake_SpikeTimes<upper_timebin));
            end
        end
        
        if trialnumber<2 %record the spikes and dwell per trial
            fake_running_speed_spikes_all_trials = fake_Running_Speed_Spikes;
       
        else
            fake_running_speed_spikes_all_trials = fake_running_speed_spikes_all_trials + fake_Running_Speed_Spikes;
            
        end
        
        
        %%
        %         %% CALCULATING ISIs ONLY WHEN HD IS WITHIN PFD RANGE OF CELL
        %
        %         % This trial ISIs
        %         this_trial_SpikeTimes = (round(SpikeTimes*10000))/10; %ISIs in ms
        %
        %         % Get a list of times when HD enters and exits the range of the cell
        %         % within_firing_range(index);
        %
        %         transition_times = zeros(numel(within_firing_range,1)); %allocate too much memory
        %         transition_count = 0;
        %         for index = 1:numel(within_firing_range)-1
        %             if(within_firing_range(index)~=within_firing_range(index+1))
        %                 transition_count = transition_count+1;
        %                 transition_times(transition_count) = index*20; %20ms bins, so this is transition time in ms.
        %             end
        %         end
        %
        %         transition_times = transition_times(transition_times>0);
        %
        %         % Need to change code depending on starting in the PFD of cell
        %         % also need to take into account the length of transition_times
        %
        %         odd_number = mod(numel(transition_times),2); %see if transition_times has an odd no of observations
        %
        %         if(within_firing_range(1)) %if HD starts in PFD of cell
        %             if(odd_number) %if odd number of transitions
        %                 move_in = transition_times(2:2:end-1);
        %                 move_out = transition_times(1:2:end);
        %             else
        %                 move_in = transition_times(2:2:end);
        %                 move_out = transition_times(1:2:end-1);
        %             end
        %         else %if HD starts outside PFD of cell
        %             if(odd_number)
        %                 move_in = transition_times(1:2:end);
        %                 move_out = transition_times(2:2:end-1);
        %             else
        %                 move_in = transition_times(1:2:end-1);
        %                 move_out = transition_times(2:2:end);
        %             end
        %         end
        %
        %
        %         %Now for each pair of in and out, take ISIs and add them to list of ISIs
        %         %for this cell
        %
        %         this_trial_PFD_ISI = zeros(1,1);
        %
        %         if(within_firing_range(1)) %if head starts in PFD
        %             if(move_out(end)>move_in(end)) %if head ends outside PFD
        %                 % take ISIs between move_in(n) and move_out(n+1)
        %                 for idx = 1:numel(move_in)
        %                     tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes>=move_in(idx) & this_trial_SpikeTimes<=move_out(idx+1));
        %                     if(numel(tempSTs>1))
        %                         tempISI = diff(tempSTs);
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI); %take ISIs in this region
        %                     end
        %
        %                     if(numel(tempSTs)) %if there's values
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,(tempSTs(1) - move_in(idx))); %take diff between first Spike in interval and start of PFD
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,move_out(idx+1) - tempSTs(end)); %take diff between end and last Spike in interval
        %                     end
        %                 end
        %                 %add in ISIs between 0 and move_out(1)
        %                 tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes<=move_out(1));
        %                 if(numel(tempSTs>1))
        %                     tempISI = diff(tempSTs);
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI);
        %                 end
        %                 if(numel(tempSTs)) %if there's values
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempSTs(1)); %take diff between first spike and start of simulation (= to first spike time)
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,move_out(1) - tempSTs(end)); %take diff between end and last Spike in interval
        %                 end
        %             else %if head ends inside PFD
        %                 %take ISIs between move
        %                 for idx = 1:numel(move_out)-1 % if head ends inside PFD
        %                     tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes>=move_in(idx) & this_trial_SpikeTimes<=move_out(idx+1));
        %                     if(numel(tempSTs>1))
        %                         tempISI = diff(tempSTs);
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI); %take ISIs in this region
        %                     end
        %                     if(numel(tempSTs)) %if there's values
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempSTs(1) - move_in(idx)); %take diff between first Spike in interval and start of PFD
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,move_out(idx+1) - tempSTs(end)); %take diff between end and last Spike in interval
        %                     end
        %                 end
        %                 %add in ISIs for the last period inside the PFD of the cell
        %                 %(between move_in(end) and end of simulation)
        %                 tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes>=move_in(end));
        %                 if(numel(tempSTs>1))
        %                     tempISI = diff(tempSTs);
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI); %take ISIs in this region
        %                 end
        %                 if(numel(tempSTs)) %if there's values
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempSTs(1) - move_in(end)); %take diff between first Spike in interval and start of PFD
        %                 end
        %
        %             end
        %
        %         else %if head starts outside PFD of the cell
        %             if(move_out(end)>move_in(end)) %if head ends outside PFD
        %                 for idx = 1:numel(move_in)
        %                     tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes>=move_in(idx) & this_trial_SpikeTimes<=move_out(idx));
        %                     if(numel(tempSTs>1))
        %                         tempISI = diff(tempSTs);
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI); %take ISIs in this region
        %                     end
        %                     if(numel(tempSTs)) %if there's values
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempSTs(1) - move_in(idx)); %take diff between first Spike in interval and start of PFD
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,move_out(idx) - tempSTs(end)); %take diff between end and last Spike in interval
        %                     end
        %                 end
        %             else %if head ends inside PFD
        %                 for idx = 1:numel(move_in)-1
        %                     tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes>=move_in(idx) & this_trial_SpikeTimes<=move_out(idx));
        %                     if(numel(tempSTs>1))
        %                         tempISI = diff(tempSTs);
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI); %take ISIs in this region
        %                     end
        %                     if(numel(tempSTs)) %if there's values
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempSTs(1) - move_in(idx)); %take diff between first Spike in interval and start of PFD
        %                         this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,move_out(idx) - tempSTs(end)); %take diff between end and last Spike in interval
        %                     end
        %                 end
        %                 %Also take ISIs for the last time head is within PFD of cell
        %                 tempSTs = this_trial_SpikeTimes(this_trial_SpikeTimes>=move_in(end));
        %                 if(numel(tempSTs>1))
        %                     tempISI = diff(tempSTs);
        %                 end
        %                 if(numel(tempSTs)) %if there's values
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempISI); %take ISIs in this region
        %                     this_trial_PFD_ISI = cat(1,this_trial_PFD_ISI,tempSTs(1) - move_in(end)); %take diff between first Spike in interval and start of PFD
        %                 end
        %             end
        %
        %         end
        %
        %         this_session_ISIs = cat(1,this_session_ISIs,this_trial_PFD_ISI);
        %         ISI_spikecount = cat(1,ISI_spikecount,numel(this_trial_PFD_ISI));
        %         filename = [trial_labels{trialnumber},'PFD_ISI.mat'];
        %         cd(savepath);
        %         save(filename, 'this_trial_PFD_ISI');
        %
        %         %% PLOT Trial ISIs
        %
        %         % FIT EXPONENTIAL
        %         TwoMsEdges = 0:2:10000;
        %         [ISIbincounts,~] = histcounts(this_trial_PFD_ISI,TwoMsEdges); %bin counts
        %         [~,ISIsortIndex] = sort(ISIbincounts(:),'descend');
        %         %ISIsortIndex(1) is the bin # where the max occurs
        %
        %         this_cell_max = TwoMsEdges(ISIsortIndex(1)) +1; %bin mean at maximum
        %
        %         %Only looking at the peak of ISI histogram to 1000ms after the peak
        %         x = 1:2:10000-1;      %bin centers
        %         x = x(ISIsortIndex(1):ISIsortIndex(1)+500)';              %transposed to column for fit
        %         y = ISIbincounts(ISIsortIndex(1):ISIsortIndex(1)+500)';   %transposed to column for it
        %
        %         if(numel(this_trial_PFD_ISI)>145) %if cell passes the criterion for inclusion in analysis
        %             %Fit exponential decay to ISI histogram from peak onwards
        %              f = fit(x,y,'exp1');
        %             coeffvals = coeffvalues(f);
        %
        %             fitted_line = coeffvals(1)*exp(coeffvals(2)*x); %exponential decay fitted
        %
        %             half_max_value = fitted_line(1)/2;
        %
        %             half_max_bin = find(fitted_line<half_max_value,1);%find where fitted_line reaches this value
        %
        %
        %
        %             [bin_counts_per_trial(trialnumber,:),~] = histcounts(this_trial_PFD_ISI,0:2:1000);
        %
        %             if ~half_max_value
        %                 %pause;
        %                 this_cell_max = NaN;
        %                 half_max_bin = NaN;
        %                 bin_counts_per_trial(trialnumber,:) = NaN;
        %                 coeffvals(1) = NaN;
        %                 coeffvals(2) = NaN;
        %             end
        %
        %         else
        %             this_cell_max = NaN;
        %             half_max_bin = NaN;
        %             bin_counts_per_trial(trialnumber,:) = NaN;
        %             coeffvals(1) = NaN;
        %             coeffvals(2) = NaN;
        %         end
        %
        %         %Get fit params for this cell
        %
        %         ISI_peak = cat(1,ISI_peak,this_cell_max);
        %         ISI_half_peak = cat(1,ISI_half_peak,half_max_bin*2);
        %         ISI_constant_a = cat(1,ISI_constant_a,coeffvals(1));
        %         ISI_constant_b = cat(1,ISI_constant_b,coeffvals(2));
        %
        %
        %         %Record fit params for this session
        %         if isempty(half_max_bin) %if it couldn't fit a time to half peak
        %             session_ISI_peaks(trialnumber,1) = NaN;
        %         else
        %             session_ISI_peaks(trialnumber,1) = half_max_bin*2; %in format, half peak time onwards from peak, peak time
        %         end
        %         session_ISI_peaks(trialnumber,2) = this_cell_max;
        %         session_ISI_exp_params(trialnumber,1) = coeffvals(1); %in format coeff1 coeff2
        %         session_ISI_exp_params(trialnumber,2) = coeffvals(2);
        %
        %         %PLOT ISIs histogram for this trial
        %         bincounts_for_histogram = histcounts(this_trial_PFD_ISI,0:2:1000);
        %         bar(bincounts_for_histogram);
        %         xlabel('Time (ms)');
        %         ylabel('Spike Count');
        %         saveas(gcf,[trial_labels{trialnumber},'_ISIs'],'jpg');
        %
        %
        %
        %         %Areas ISIs
        %         curr_area = this_session_trials{trialnumber};
        %         curr_area = curr_area(1:3);
        %         if(strcmp(curr_area,'RSC'))
        %             RSC_count = RSC_count+1;
        %             if isempty(half_max_bin) %if it couldn't fit a time to half peak
        %                 RSC_ISI_half_peaks(RSC_count) = NaN;
        %             else
        %                 RSC_ISI_half_peaks(RSC_count) = half_max_bin*2;
        %             end
        %             RSC_ISI_peaks(RSC_count) = this_cell_max;
        %             RSC_ISI_constant_a(RSC_count) = coeffvals(1);
        %             RSC_ISI_constant_b(RSC_count) = coeffvals(2);
        %
        %             RSC_ISIs = cat(1,RSC_ISIs,this_trial_PFD_ISI);
        %
        %             RSC_trial_names{RSC_count} = ['r',num2str(rats(session)),'_',dates{session},'_'...
        %                 ,trial_labels{trialnumber},',',tetrodes(session),',',clusters(session)];
        %
        %         else
        %             POS_count = POS_count+1;
        %
        %             if isempty(half_max_bin) %if it couldn't fit a time to half peak
        %                 POS_ISI_half_peaks(POS_count) = NaN;
        %             else
        %                 POS_ISI_half_peaks(POS_count) = half_max_bin*2;
        %             end
        %             POS_ISI_peaks(POS_count) = this_cell_max;
        %
        %             POS_ISI_constant_a(POS_count) = coeffvals(1);
        %             POS_ISI_constant_b(POS_count) = coeffvals(2);
        %
        %             POS_ISIs = cat(1,POS_ISIs,this_trial_PFD_ISI);
        %
        %             POS_trial_names{POS_count} = ['r',num2str(rats(session)),'_',dates{session},'_'...
        %                 ,trial_labels{trialnumber},',',tetrodes(session),',',clusters(session)];
        %
        %         end
        %
        %         %%
        %         %%%%%%%%%%%%%%% Speed-Theta Analysis (codes from Giulio/Burgess)
        %
        %         [ Stats] = ThetaAnalysisFW(mtint,1,trial_labels{trialnumber},savepath,num2str(rats(session)),dates{session},1);
        %
        %
        %
        %         LFP_frequency(trialnumber,1) = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.maxFreq;
        %         LFP_frequency(trialnumber,2) = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.bandMaxPower;
        %
        %         frequencies = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.freqs;
        %         powers = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.power;
        %
        %
        %         delta_mask = frequencies>=1 & frequencies <=4;
        %         theta_mask = frequencies>=7 & frequencies <=11;
        %         delta_powers = powers(delta_mask);
        %         mean_delta = mean(delta_powers);
        %         theta_powers = powers(theta_mask);
        %         mean_theta = mean(theta_powers);
        %
        %
        %         theta_ratio(trialnumber,1) = mean_theta/mean_delta;
        %
        %
        %         %other version below:
        %         alt_theta_ratio(trialnumber,1) = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.bandMaxPower/mean_delta;
        %
        %
        %         % Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.freqs is the
        %         % frequencies
        %
        %         %Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.power is the
        %         %power at these frequencies
        %
        %
        %         speed_theta_stats(trialnumber,1) = Stats.mtint.theta.hilBeta; %slope of speed-theta
        %         speed_theta_stats(trialnumber,2) =  Stats.mtint.theta.hilF0; %interecept of speed-theta
        %         speed_theta_stats(trialnumber,3) =  Stats.mtint.theta.r_speed_theta_binned_data; %r value of speed-theta
        %         speed_theta_stats(trialnumber,4) =  Stats.mtint.theta.p_speed_theta_binned_data; %p value of speed-theta
        %
        %
        %         %%
        %         %%%%%%%%%%%%%%% Intrinsic frequency analysis (codes from Giulio/Burgess)
        %         Vars=daVarsStruct;
        %         max_speed = Vars.pos.maxSpeedThetaBin;
        %         NBinsOfSpikingThetaModulation = Vars.eeg.NBinsOfSpikingThetaModulation;
        %         SpeedBinTheta    = (max_speed - 0) / NBinsOfSpikingThetaModulation ;%Vars.eeg.speedBinLengthCm;S
        %         Ranges = 0 : SpeedBinTheta : max_speed;Ranges(find(Ranges<Vars.pos.minSpeed)) = [] ;
        %         HalfBin = diff(Ranges) ;HalfBin=HalfBin(1)/2;
        %         BinCentres = Ranges + HalfBin ;
        %         BinCentres(find(BinCentres > max_speed)) = [] ;
        %         SpeedRange={'All'};
        %
        %         for iSpeed = 1 :   numel(BinCentres)
        %             SpeedRange{iSpeed+1} = [num2str(BinCentres(iSpeed))] ;
        %         end
        %         clear iSpeed;
        %
        %
        %         MinPossibleSpeed = 0;%Vars.pos.minSpeed;
        %         speed = mtint.pos.speed ;
        %         speed(find(speed < MinPossibleSpeed)) = NaN;
        %         speed(find(speed > max_speed)) = NaN;
        %         acBinSize=0.002;                                                                     in.acBinSize =acBinSize;
        %         acWindow =0.5;                                                                       in.acWindow=acWindow;
        %         thetaRange=Vars.eeg.psTh;                                                            in.thetaRange=thetaRange;
        %         padToPow2=16;                                                                        in.padToPow2=padToPow2;
        %         smthKernelWidth=Vars.eeg.psSmth(1);                                                  in.smthKernelWidth=smthKernelWidth;
        %         smthKernelSigma=Vars.eeg.psSmth(2);                                                  in.smthKernelSigma=smthKernelSigma;
        %         maxFreq=Vars.eeg.psMaxF;                                                             in.maxFreq=maxFreq;
        %         s2nWidth = 2;                                                                        in.s2nWidth=s2nWidth;
        %         ymax =[];                                                                            in.ymax=ymax;
        %         PLOT_ON= 1;                                                                          in.PLOT_ON=PLOT_ON;
        %         posMask = ones(length(speed) ,1);                                                    in.posMask=posMask;
        %         in.spikeTimes=SpikeTimes;
        %         %%%% Now running function
        %         ret = intrinsic_freq_autoCorr(in);
        %
        %         intrinsic_freq_stats(trialnumber,1) = ret.maxFreq;
        %         intrinsic_freq_stats(trialnumber,2) = ret.bandMaxPower;
        %
        %         frequencies = ret.freqs;
        %         powers = ret.power;
        %         delta_mask = frequencies>=1 & frequencies <=4;
        %         theta_mask = frequencies>=7 & frequencies <=11;
        %         delta_powers = powers(delta_mask);
        %         mean_delta = mean(delta_powers);
        %         theta_powers = powers(theta_mask);
        %         mean_theta = mean(theta_powers);
        %         intrinsic_freq_stats(trialnumber,3) = mean_theta/mean_delta;
        %         intrinsic_freq_stats(trialnumber,4) = ret.bandMaxPower/mean_delta;
        %
        %         %saving power spectrum
        %         intrinsic_title = ['Session_', trial_labels{trialnumber}, '_',num2str(rats(session)) ,'_', dates{session},'_IntrinsicTheta'];
        %         intrinsic_title = strrep(intrinsic_title, '.', '_');
        %         saveas(gcf,intrinsic_title,'tif');
        %         close(gcf);
        %         %%%%%%%%%%%%%%
        fclose('all');
        
        cd(startpath); %GO HOME
        
    end
    
    %%Getting average max rate
    avg_max_rate(session) = mean(max_rate(:)); %mean of max rate across all trials
    
    %%Now doing the fits etc. on the data summarised across all trials within a given session
    cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/NEW_Summary_plots_for_Kate/per_session_analysis'); %save in the plots directory
    %% Running speed
    figure()
    subplot(2,1,1)  %regular version
    running_speed_rates = running_speed_spikes_all_trials./running_speed_dwell_all_trials;
    
    %correct for low dwell times
    bin_exclusion_criterion = 1.5;  %0.5s% of the time %changed to 1.5s from 0.5s
    bin_exclusion_filter = running_speed_dwell_all_trials<bin_exclusion_criterion & running_speed_spikes_all_trials<5; %changed from 2 to 5
    
    running_speed_rates(bin_exclusion_filter) = NaN;
    
    
    %now doing regression between firing rates and the bin mean
    bin_centers = zeros(numel(running_speed_rates),1);
    for index = 1:numel(running_speed_rates)
        bin_centers(index) = (speed_binedges(index) + speed_binedges(index+1))/2;
    end
    x = bin_centers(~bin_exclusion_filter);                %Speed is x
    y = running_speed_rates(~bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    
    %fit a regression line to scatter plot
    idx = isnan(running_speed_rates);
    
    if numel(running_speed_rates(~bin_exclusion_filter))>=10 %only if sufficient points to fit
        format long
        b = polyfit(bin_centers(~idx), running_speed_rates(~idx), 1)';
        hold on
        plot(bin_centers,polyval(b,bin_centers),'r');
        yCalc = polyval(b,bin_centers);
        yCalc = yCalc(~bin_exclusion_filter);
        R2 = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);
        
        %Now exponential fitting
        f = fit(x,y,'exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*bin_centers); %exponential decay fitted
        plot(bin_centers,fitted_line,'g');
        actual_fit = fitted_line(~bin_exclusion_filter);
        R2_exp = 1 - sum((y - actual_fit).^2)/sum((y - mean(y)).^2);
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    
    xlabel('Running Speed (cm/s)');
    ylabel('Firing Rate (Hz)');
    
    ylim([0 inf]); %set just lower to be 0Hz
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    
    running_speed_stats_all_sessions(session,:) =[b(1),b(2),(R2),lin_corr]; %slope intercept R^2 r
    running_speed_exp_stats_all_sessions(session,:) = [coeffvals(1), coeffvals(2), R2_exp]; %a b R^2
    
    subplot(2,1,2) 
    %shuffled version
    
    running_speed_slopes = zeros(5000,1);
    original_running_speed_spikes_all_trials = running_speed_spikes_all_trials;
    for count = 1:5000
        count
        running_speed_spikes_all_trials = original_running_speed_spikes_all_trials;
    running_speed_spikes_all_trials(bin_exclusion_filter) = NaN;
    shuffled_running_speed_spikes_all_trials = nanshuffle(running_speed_spikes_all_trials)';
    shuffled_running_speed_rates = shuffled_running_speed_spikes_all_trials./running_speed_dwell_all_trials; %dwell not shuffled anyway same as normal
    
    bin_exclusion_filter = isnan(shuffled_running_speed_rates);
    
    %now doing regression between firing rates and the bin mean
    bin_centers = zeros(numel(shuffled_running_speed_rates),1);
    for index = 1:numel(shuffled_running_speed_rates)
        bin_centers(index) = (speed_binedges(index) + speed_binedges(index+1))/2;
    end
    x = bin_centers(~bin_exclusion_filter);                %Speed is x
    y = shuffled_running_speed_rates(~bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
   % scatter(x,y,'.');
    
    %fit a regression line to scatter plot
    idx = isnan(shuffled_running_speed_rates);
    
    if numel(shuffled_running_speed_rates(~bin_exclusion_filter))>=10 %only if sufficient points to fit
        format long
        b = polyfit(bin_centers(~idx), shuffled_running_speed_rates(~idx), 1)';
        hold on
        %plot(bin_centers(~idx),polyval(b,bin_centers(~idx)),'r');
        yCalc = polyval(b,bin_centers(~idx));
        %yCalc = yCalc(~bin_exclusion_filter);
        R2 = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);
        
        %Now exponential fitting
        f = fit(x,y,'exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*bin_centers(~idx)); %exponential decay fitted
        %plot(bin_centers(~idx),fitted_line,'g');
        actual_fit = fitted_line;
        R2_exp = 1 - sum((y - actual_fit).^2)/sum((y - mean(y)).^2);
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    running_speed_slopes(count) = b(1);
    end
    xlabel('Running Speed (cm/s)');
    ylabel('Firing Rate (Hz)');
    ylim([0 inf]); %set just lower to be 0Hz
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    new_session_text = strrep(session_text{session},'.','');
    new_session_text = strrep(new_session_text,'\','_');
    %saveas(gcf,[ new_session_text,'_running_speed'],'tif');
    close(gcf);
    
    shuffled_running_speed_stats_all_sessions(session,:) =[b(1),b(2),(R2),lin_corr]; %slope intercept R^2 r
    shuffled_running_speed_exp_stats_all_sessions(session,:) = [coeffvals(1), coeffvals(2), R2_exp]; %a b R^2
    
    %% R turns
    
    R_turn_rates = R_turn_spikes_all_trials./R_turn_dwell_all_trials; %correct for amount of time spent at different speeds
    
    %get rid of undersampled AHV
    total_R_AHV_dwell = sum(R_turn_dwell_all_trials);  %total time spent
    R_AHV_bin_exclusion_criterion = total_R_AHV_dwell/600;  %0.5% of the time %changed 200 to 600 to get 1.5% of time now
    R_AHV_bin_exclusion_filter = R_turn_dwell_all_trials>R_AHV_bin_exclusion_criterion & R_turn_spikes_all_trials>=5; %changed 2 to 5
    
    %get bin centers - can use same bins for R and L
    R_AHV_bin_centers = zeros(numel(R_turn_rates),1);
    for index = 1:numel(R_turn_rates)
        R_AHV_bin_centers(index) = (R_AHV_binedges(index) + R_AHV_binedges(index+1))/2;
    end
    
    
    figure();
    subplot(2,1,1);
    %NOW GET STATS FOR R
    x = R_AHV_bin_centers(R_AHV_bin_exclusion_filter);                 %AHV is x
    y = R_turn_rates(R_AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    R_turn_rates(~R_AHV_bin_exclusion_filter) = NaN;
    idx = isnan(R_turn_rates);
    format long
    
    if numel(R_turn_rates(R_AHV_bin_exclusion_filter))>=10 %if there's enough points to fit
        b = polyfit(R_AHV_bin_centers(~idx), R_turn_rates(~idx)', 1)';
        hold on
        plot(R_AHV_bin_centers(R_AHV_bin_exclusion_filter),polyval(b,R_AHV_bin_centers(R_AHV_bin_exclusion_filter)),'r');
        yCalc = polyval(b,R_AHV_bin_centers(R_AHV_bin_exclusion_filter));
        %yCalc = yCalc(R_AHV_bin_exclusion_filter);
        R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
        %Now exponential fitting
        f = fit(x,y','exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*R_AHV_bin_centers(R_AHV_bin_exclusion_filter)); %exponential decay fitted
        plot(R_AHV_bin_centers(R_AHV_bin_exclusion_filter),fitted_line,'g');
        actual_fit = fitted_line;
        R2_exp = 1 - sum((y' - actual_fit).^2)/sum((y' - mean(y)).^2);
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    xlabel('R Head Rotation (deg/s)');
    ylabel('Firing Rate (Hz)');
    
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    xlim([0 200]);
    ylim([0 inf]); %set just lower to be 0Hz
    
    right_turn_stats_all_sessions(session,:) = [b(1),b(2),R2,lin_corr]; %slope intercept R^2 r
    right_turn_exp_stats_all_sessions(session,:) = [coeffvals(1),coeffvals(2),R2_exp]; %a b R^2
    
    
    %now doing the shuffled version
%     shuffled_R_turn_spikes_all_trials = R_turn_spikes_all_trials;
%     shuffled_R_turn_spikes_all_trials(~R_AHV_bin_exclusion_filter) = NaN;
%     shuffled_R_turn_spikes_all_trials = shuffled_R_turn_spikes_all_trials(randperm(numel(shuffled_R_turn_spikes_all_trials))); %shuffle that badboy
    
  
        R_turn_spikes_all_trials(~R_AHV_bin_exclusion_filter) = NaN;
        shuffled_R_turn_spikes_all_trials = nanshuffle(R_turn_spikes_all_trials);
        
        shuffled_R_turn_rates = shuffled_R_turn_spikes_all_trials./R_turn_dwell_all_trials; %dwell not shuffled anyway same as normal
        
        shuffled_R_AHV_bin_exclusion_filter = ~isnan(shuffled_R_turn_rates);
        
        subplot(2,1,2);
        %NOW GET STATS FOR R
        x = R_AHV_bin_centers(shuffled_R_AHV_bin_exclusion_filter);                 %AHV is x
        y = shuffled_R_turn_rates(shuffled_R_AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
        %scatter(x,y,'.');
        shuffled_R_turn_rates(~shuffled_R_AHV_bin_exclusion_filter) = NaN;
        idx = isnan(shuffled_R_turn_rates);
        format long
        
        if numel(shuffled_R_turn_rates(shuffled_R_AHV_bin_exclusion_filter))>=10 %if there's enough points to fit
            b = polyfit(R_AHV_bin_centers(~idx), shuffled_R_turn_rates(~idx)', 1)';
            hold on
            plot(R_AHV_bin_centers(shuffled_R_AHV_bin_exclusion_filter),polyval(b,R_AHV_bin_centers(shuffled_R_AHV_bin_exclusion_filter)),'r');
            yCalc = polyval(b,R_AHV_bin_centers(shuffled_R_AHV_bin_exclusion_filter));
            yCalc = yCalc(shuffled_R_AHV_bin_exclusion_filter);
            R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
            %Now exponential fitting
            f = fit(x,y','exp1');
            coeffvals = coeffvalues(f);
            fitted_line = coeffvals(1)*exp(coeffvals(2)*R_AHV_bin_centers(shuffled_R_AHV_bin_exclusion_filter)); %exponential decay fitted
            plot(R_AHV_bin_centers(shuffled_R_AHV_bin_exclusion_filter),fitted_line,'g');
            actual_fit = fitted_line;
            R2_exp = 1 - sum((y' - actual_fit).^2)/sum((y' - mean(y)).^2);
            lin_corr = corr(x(:),y(:));
        else
            b = [NaN NaN];
            R2 = NaN;
            R2_exp = NaN;
            coeffvals = [NaN NaN];
            
            lin_corr = NaN;
        end
        xlabel('R Head Rotation (deg/s)');
        ylabel('Firing Rate (Hz)');
        
        
       
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    xlim([0 200]);
    ylim([0 inf]); %set just lower to be 0Hz
    
    %saveas(gcf,[new_session_text,'_R_turns'],'tif');
    close(gcf);
    
    shuffled_right_turn_stats_all_sessions(session,:) = [b(1),b(2),R2,lin_corr]; %slope intercept R^2 r
    shuffled_right_turn_exp_stats_all_sessions(session,:) = [coeffvals(1),coeffvals(2),R2_exp]; %a b R^2
    
    
    
    %% L turns
    
    L_turn_rates = L_turn_spikes_all_trials./L_turn_dwell_all_trials; %correct for amount of time spent at different speeds
    
    %get rid of undersampled AHV
    total_L_AHV_dwell = sum(L_turn_dwell_all_trials);  %total time spent
    L_AHV_bin_exclusion_criterion = total_L_AHV_dwell/600;  %0.5% of the time at 200 up to 1.5% of time at 600
    L_AHV_bin_exclusion_filter = L_turn_dwell_all_trials>L_AHV_bin_exclusion_criterion & L_turn_spikes_all_trials>=5; %changed 2 to 5
    
    original_L_bin_exclusion = L_AHV_bin_exclusion_filter;
    original_L_dwell = L_turn_dwell_all_trials;
    
    L_AHV_bin_centers = fliplr(R_AHV_bin_centers);
    L_turn_rates = fliplr(L_turn_rates);
    L_AHV_bin_exclusion_filter = fliplr(L_AHV_bin_exclusion_filter);
    
    %NOW GET STATS FOR L
    figure()
    subplot(2,1,1);
    x = L_AHV_bin_centers(L_AHV_bin_exclusion_filter);                 %AHV is x
    y = L_turn_rates(L_AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    L_turn_rates(~L_AHV_bin_exclusion_filter) = NaN;
    idx = isnan(L_turn_rates);
    format long
    if numel(L_turn_rates(L_AHV_bin_exclusion_filter))>=10 %only fit if enough points
        b = polyfit(L_AHV_bin_centers(~idx), L_turn_rates(~idx)', 1)';
        hold on
        plot(L_AHV_bin_centers(L_AHV_bin_exclusion_filter),polyval(b,L_AHV_bin_centers(L_AHV_bin_exclusion_filter)),'r');
        yCalc = polyval(b,L_AHV_bin_centers(L_AHV_bin_exclusion_filter));
        %yCalc = yCalc(L_AHV_bin_exclusion_filter);
        R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
        
        %Now exponential fitting
        f = fit(x,y','exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*L_AHV_bin_centers(L_AHV_bin_exclusion_filter)); %exponential decay fitted
        plot(L_AHV_bin_centers(L_AHV_bin_exclusion_filter),fitted_line,'g');
        actual_fit = fitted_line;
        R2_exp = 1 - sum((y' - actual_fit).^2)/sum((y' - mean(y)).^2);
        
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    xlabel('L Head Rotation (deg/s)');
    ylabel('Firing Rate (Hz)');
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    xlim([0 200]);
    ylim([0 inf]); %set just lower to be 0Hz
    
    left_turn_stats_all_sessions(session,:) =  [b(1),b(2),R2,lin_corr]; %slope intercept R^2 r
    left_turn_exp_stats_all_sessions(session,:) = [coeffvals(1),coeffvals(2),R2_exp]; %a b R^2
    
    %shuffled version
    
    %need the non-flipped versions first
     L_AHV_bin_exclusion_filter= original_L_bin_exclusion;
     L_turn_dwell_all_trials = original_L_dwell;

    L_turn_spikes_all_trials(~L_AHV_bin_exclusion_filter) = NaN;
    shuffled_L_turn_spikes_all_trials = nanshuffle(L_turn_spikes_all_trials);
    
    shuffled_L_turn_rates = shuffled_L_turn_spikes_all_trials./L_turn_dwell_all_trials; %dwell not shuffled anyway same as normal

    shuffled_L_AHV_bin_exclusion_filter = ~isnan(shuffled_L_turn_rates);
    
    shuffled_L_turn_rates = fliplr(shuffled_L_turn_rates);
    shuffled_L_AHV_bin_exclusion_filter = fliplr(shuffled_L_AHV_bin_exclusion_filter);
    
    subplot(2,1,2);
    x = L_AHV_bin_centers(shuffled_L_AHV_bin_exclusion_filter);                 %AHV is x
    y = shuffled_L_turn_rates(shuffled_L_AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    shuffled_L_turn_rates(~shuffled_L_AHV_bin_exclusion_filter) = NaN;
    idx = isnan(shuffled_L_turn_rates);
    format long
    if numel(shuffled_L_turn_rates(shuffled_L_AHV_bin_exclusion_filter))>=10 %only fit if enough points
        b = polyfit(L_AHV_bin_centers(~idx), shuffled_L_turn_rates(~idx)', 1)';
        hold on
        plot(L_AHV_bin_centers(shuffled_L_AHV_bin_exclusion_filter),polyval(b,L_AHV_bin_centers(shuffled_L_AHV_bin_exclusion_filter)),'r');
        yCalc = polyval(b,L_AHV_bin_centers(shuffled_L_AHV_bin_exclusion_filter));
        %yCalc = yCalc(shuffled_L_AHV_bin_exclusion_filter);
        R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
       
        %Now exponential fitting
        f = fit(x,y','exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*L_AHV_bin_centers(shuffled_L_AHV_bin_exclusion_filter)); %exponential decay fitted
        plot(L_AHV_bin_centers(shuffled_L_AHV_bin_exclusion_filter),fitted_line,'g');
        actual_fit = fitted_line;
        R2_exp = 1 - sum((y' - actual_fit).^2)/sum((y' - mean(y)).^2);
        
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    xlabel('L Head Rotation (deg/s)');
    ylabel('Firing Rate (Hz)');
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    xlim([0 200]);
    ylim([0 inf]); %set just lower to be 0Hz
    
    %saveas(gcf,[new_session_text,'_L_turns'],'tif');
    close(gcf);
    
    shuffled_left_turn_stats_all_sessions(session,:) = [b(1),b(2),R2,lin_corr]; %slope intercept R^2 r
    shuffled_left_turn_exp_stats_all_sessions(session,:) = [coeffvals(1),coeffvals(2),R2_exp]; %a b R^2
    
    
    
    %% Now doing the FAKE spike trains
    
    % R turns
    R_turn_rates = fake_R_turn_spikes_all_trials./R_turn_dwell_all_trials; %correct for amount of time spent at different speeds
    
    %get rid of undersampled AHV
    total_R_AHV_dwell = sum(R_turn_dwell_all_trials);  %total time spent
    R_AHV_bin_exclusion_criterion = total_R_AHV_dwell/600;  %0.5% of the time at 200 up to 1.5% of time at 600
    R_AHV_bin_exclusion_filter = R_turn_dwell_all_trials>R_AHV_bin_exclusion_criterion & fake_R_turn_spikes_all_trials>=5; %changed 2 to 5
    
    %get bin centers - can use same bins for R and L
    R_AHV_bin_centers = zeros(numel(R_turn_rates),1);
    for index = 1:numel(R_turn_rates)
        R_AHV_bin_centers(index) = (R_AHV_binedges(index) + R_AHV_binedges(index+1))/2;
    end
    
    
    figure();
    subplot(2,1,1);
    %NOW GET STATS FOR R
    x = R_AHV_bin_centers(R_AHV_bin_exclusion_filter);                 %AHV is x
    y = R_turn_rates(R_AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    R_turn_rates(~R_AHV_bin_exclusion_filter) = NaN;
    idx = isnan(R_turn_rates);
    format long
    
    if numel(R_turn_rates(R_AHV_bin_exclusion_filter))>=10 %if there's enough points to fit
        b = polyfit(R_AHV_bin_centers(~idx), R_turn_rates(~idx)', 1)';
        hold on
        plot(R_AHV_bin_centers(R_AHV_bin_exclusion_filter),polyval(b,R_AHV_bin_centers(R_AHV_bin_exclusion_filter)),'r');
        yCalc = polyval(b,R_AHV_bin_centers(R_AHV_bin_exclusion_filter));
        %yCalc = yCalc(R_AHV_bin_exclusion_filter);
        R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
        %Now exponential fitting
        f = fit(x,y','exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*R_AHV_bin_centers(R_AHV_bin_exclusion_filter)); %exponential decay fitted
        plot(R_AHV_bin_centers(R_AHV_bin_exclusion_filter),fitted_line,'g');
        actual_fit = fitted_line;
        R2_exp = 1 - sum((y' - actual_fit).^2)/sum((y' - mean(y)).^2);
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    xlabel('R Head Rotation (deg/s)');
    ylabel('Firing Rate (Hz)');
    
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    xlim([0 200]);
    ylim([0 inf]); %set just lower to be 0Hz
    
    fake_right_turn_stats_all_sessions(session,:) = [b(1),b(2),R2,lin_corr]; %slope intercept R^2 r
    fake_right_turn_exp_stats_all_sessions(session,:) = [coeffvals(1),coeffvals(2),R2_exp]; %a b R^2
    
    L_turn_rates = fake_L_turn_spikes_all_trials./L_turn_dwell_all_trials; %correct for amount of time spent at different speeds
    
    %get rid of undersampled AHV
    total_L_AHV_dwell = sum(L_turn_dwell_all_trials);  %total time spent
    L_AHV_bin_exclusion_criterion = total_L_AHV_dwell/600;  %0.5% of the time at 200 up to 1.5% of time at 600
    L_AHV_bin_exclusion_filter = L_turn_dwell_all_trials>L_AHV_bin_exclusion_criterion & fake_L_turn_spikes_all_trials>=5; %changed 2 to 5
    
    L_AHV_bin_centers = fliplr(R_AHV_bin_centers);
    L_turn_rates = fliplr(L_turn_rates);
    L_AHV_bin_exclusion_filter = fliplr(L_AHV_bin_exclusion_filter);
    
    %NOW GET STATS FOR L
    subplot(2,1,2);
    x = L_AHV_bin_centers(L_AHV_bin_exclusion_filter);                 %AHV is x
    y = L_turn_rates(L_AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    L_turn_rates(~L_AHV_bin_exclusion_filter) = NaN;
    idx = isnan(L_turn_rates);
    format long
    if numel(L_turn_rates(L_AHV_bin_exclusion_filter))>=10 %only fit if enough points
        b = polyfit(L_AHV_bin_centers(~idx), L_turn_rates(~idx)', 1)';
        hold on
        plot(L_AHV_bin_centers(L_AHV_bin_exclusion_filter),polyval(b,L_AHV_bin_centers(L_AHV_bin_exclusion_filter)),'r');
        yCalc = polyval(b,L_AHV_bin_centers(L_AHV_bin_exclusion_filter));
        %yCalc = yCalc(L_AHV_bin_exclusion_filter);
        R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
        
        %Now exponential fitting
        f = fit(x,y','exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*L_AHV_bin_centers(L_AHV_bin_exclusion_filter)); %exponential decay fitted
        plot(L_AHV_bin_centers(L_AHV_bin_exclusion_filter),fitted_line,'g');
        actual_fit = fitted_line;
        R2_exp = 1 - sum((y' - actual_fit).^2)/sum((y' - mean(y)).^2);
        
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    xlabel('L Head Rotation (deg/s)');
    ylabel('Firing Rate (Hz)');
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    xlim([0 200]);
    ylim([0 inf]); %set just lower to be 0Hz    
    
    fake_left_turn_stats_all_sessions(session,:) = [b(1),b(2),R2,lin_corr]; %slope intercept R^2 r
    fake_left_turn_exp_stats_all_sessions(session,:) = [coeffvals(1),coeffvals(2),R2_exp]; %a b R^2
    
    saveas(gcf,[new_session_text,'_fake_turns'],'tif');
    close(gcf);
    
    %% Fake running speed too
     figure()
    fake_running_speed_rates = fake_running_speed_spikes_all_trials./running_speed_dwell_all_trials;
    
    %correct for low dwell times
    bin_exclusion_criterion = 1.5;  %0.5s% of the time %changed to 1.5s from 0.5s
    bin_exclusion_filter = running_speed_dwell_all_trials<bin_exclusion_criterion & fake_running_speed_spikes_all_trials<5; %changed from 2 to 5
    
    fake_running_speed_rates(bin_exclusion_filter) = NaN;
    
    
    %now doing regression between firing rates and the bin mean
    bin_centers = zeros(numel(fake_running_speed_rates),1);
    for index = 1:numel(fake_running_speed_rates)
        bin_centers(index) = (speed_binedges(index) + speed_binedges(index+1))/2;
    end
    x = bin_centers(~bin_exclusion_filter);                %Speed is x
    y = fake_running_speed_rates(~bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    scatter(x,y,'.');
    
    %fit a regression line to scatter plot
    idx = isnan(fake_running_speed_rates);
    
    if numel(fake_running_speed_rates(~bin_exclusion_filter))>=10 %only if sufficient points to fit
        format long
        b = polyfit(bin_centers(~idx), fake_running_speed_rates(~idx), 1)';
        hold on
        plot(bin_centers,polyval(b,bin_centers),'r');
        yCalc = polyval(b,bin_centers);
        yCalc = yCalc(~bin_exclusion_filter);
        R2 = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);
        
        %Now exponential fitting
        f = fit(x,y,'exp1');
        coeffvals = coeffvalues(f);
        fitted_line = coeffvals(1)*exp(coeffvals(2)*bin_centers); %exponential decay fitted
        plot(bin_centers,fitted_line,'g');
        actual_fit = fitted_line(~bin_exclusion_filter);
        R2_exp = 1 - sum((y - actual_fit).^2)/sum((y - mean(y)).^2);
        lin_corr = corr(x(:),y(:));
    else
        b = [NaN NaN];
        R2 = NaN;
        R2_exp = NaN;
        coeffvals = [NaN NaN];
        
        lin_corr = NaN;
    end
    
    xlabel('Running Speed (cm/s)');
    ylabel('Firing Rate (Hz)');
    
    ylim([0 inf]); %set just lower to be 0Hz
    
    title([' Slope: ',num2str(b(1)),' Intercept: ',num2str(b(2)),' R^2', num2str(R2),' R^2 (exp):',num2str(R2_exp)]);
    
    fake_running_speed_stats_all_sessions(session,:) =[b(1),b(2),(R2),lin_corr]; %slope intercept R^2 r
    fake_running_speed_exp_stats_all_sessions(session,:) = [coeffvals(1), coeffvals(2), R2_exp]; %a b R^2
    
     saveas(gcf,[new_session_text,'_fake_running_speed'],'tif');
     close(gcf);
    
    
    %     %saving overall ISIs
    %     save('this_session_ISIs.mat', 'this_session_ISIs');
    %
    %     %saving bin counts on a per-cell basis
    %     save('bin_counts_per_trial.mat','bin_counts_per_trial');
    %
    %     % saving fit parameters
    %
    %     save('ISI_peak.mat','ISI_peak');
    %     save('ISI_half_peak.mat','ISI_half_peak');
    %     save('ISI_constant_a.mat','ISI_constant_a');
    %     save('ISI_constant_b.mat','ISI_constant_b');
    %     save('ISI_spikecount.mat','ISI_spikecount');
    %
    %     save('session_ISI_peaks.mat','session_ISI_peaks');
    %     save('session_ISI_exp_params.mat','session_ISI_exp_params');
    %
    %
    %     %LFP theta
    %     save('speed_theta_stats.mat','speed_theta_stats');
    %     save('LFP_frequency.mat','LFP_frequency');
    %     save('theta_ratio_all.mat','theta_ratio');
    %     save('theta_alt_ratio_all.mat','alt_theta_ratio');
    %     save('corrected_intrinsic_freq_stats.mat', 'intrinsic_freq_stats');
    %
    cd(startpath);
    
    %clear all the variables for this session
    
    %     clear shuffled_running_speed_stats shuffled_AHV_stats shuffled_absAHV_stats shuffled_running_speed_r shuffled_AHV_r shuffled_absAHV_r this_session_ISIs bin_counts_per_trial ISI_half_peak ISI_constant_a ISI_constant_b ISI_spikecount...
    %         session_ISI_peaks session_ISI_exp_params speed_theta_stats LFP_frequency theta_ratio alt_theta_ratio intrinsic_freq_stats ISI_peak AHV_distirbutions absAVH distributions running_speed_distributions
    
end


%% SAVE AREA STATS
cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis');


%% saving all the stats for collapsed sessions
if strcmpi(mode,'first')
%     save('running_speed_stats_all_sessions_1.mat','running_speed_stats_all_sessions');
%     save('shuffled_running_speed_stats_all_sessions_1.mat','shuffled_running_speed_stats_all_sessions');
%     save('running_speed_exp_stats_all_sessions_1.mat','running_speed_exp_stats_all_sessions');
%     save('shuffled_running_speed_exp_stats_all_sessions_1.mat','shuffled_running_speed_exp_stats_all_sessions');
%     
%     save('left_turn_stats_all_sessions_1.mat','left_turn_stats_all_sessions');
%     save('shuffled_left_turn_stats_all_sessions_1.mat','shuffled_left_turn_stats_all_sessions');
%     save('left_turn_exp_stats_all_sessions_1.mat','left_turn_exp_stats_all_sessions');
%     save('shuffled_left_turn_exp_stats_all_sessions_1.mat','shuffled_left_turn_exp_stats_all_sessions');
%     
%     save('right_turn_stats_all_sessions_1.mat','right_turn_stats_all_sessions');
%     save('shuffled_right_turn_stats_all_sessions_1.mat','shuffled_right_turn_stats_all_sessions');
%     save('right_turn_exp_stats_all_sessions_1.mat','right_turn_exp_stats_all_sessions');
%     save('shuffled_right_turn_exp_stats_all_sessions_1.mat','shuffled_right_turn_exp_stats_all_sessions');
%     
%     save('fake_left_turn_stats_all_sessions_1.mat','fake_left_turn_stats_all_sessions');
%     save('fake_left_turn_exp_stats_all_sessions_1.mat','fake_left_turn_exp_stats_all_sessions');
%     save('fake_right_turn_stats_all_sessions_1.mat','fake_right_turn_stats_all_sessions');
%     save('fake_right_turn_exp_stats_all_sessions_1.mat','fake_right_turn_exp_stats_all_sessions');
%     
%     
%     save('avg_max_rate_1.mat','avg_max_rate');

    save('fake_running_speed_exp_stats_all_sessions_1.mat','fake_running_speed_exp_stats_all_sessions');
    save('fake_running_speed_stats_all_sessions_1.mat','fake_running_speed_stats_all_sessions');
elseif strcmpi(mode,'second')
%     save('running_speed_stats_all_sessions_2.mat','running_speed_stats_all_sessions');
%     save('shuffled_running_speed_stats_all_sessions_2.mat','shuffled_running_speed_stats_all_sessions');
%     save('running_speed_exp_stats_all_sessions_2.mat','running_speed_exp_stats_all_sessions');
%     save('shuffled_running_speed_exp_stats_all_sessions_2.mat','shuffled_running_speed_exp_stats_all_sessions');
%     
%     save('left_turn_stats_all_sessions_2.mat','left_turn_stats_all_sessions');
%     save('shuffled_left_turn_stats_all_sessions_2.mat','shuffled_left_turn_stats_all_sessions');
%     save('left_turn_exp_stats_all_sessions_2.mat','left_turn_exp_stats_all_sessions');
%     save('shuffled_left_turn_exp_stats_all_sessions_2.mat','shuffled_left_turn_exp_stats_all_sessions');
%     
%     save('right_turn_stats_all_sessions_2.mat','right_turn_stats_all_sessions');
%     save('shuffled_right_turn_stats_all_sessions_2.mat','shuffled_right_turn_stats_all_sessions');
%     save('right_turn_exp_stats_all_sessions_2.mat','right_turn_exp_stats_all_sessions');
%     save('shuffled_right_turn_exp_stats_all_sessions_2.mat','shuffled_right_turn_exp_stats_all_sessions');
%     
%     
%     save('fake_left_turn_stats_all_sessions_2.mat','fake_left_turn_stats_all_sessions');
%     save('fake_left_turn_exp_stats_all_sessions_2.mat','fake_left_turn_exp_stats_all_sessions');
%     save('fake_right_turn_stats_all_sessions_2.mat','fake_right_turn_stats_all_sessions');
%     save('fake_right_turn_exp_stats_all_sessions_2.mat','fake_right_turn_exp_stats_all_sessions');
%     
%     
%     save('avg_max_rate_2.mat','avg_max_rate');
    
    save('fake_running_speed_exp_stats_all_sessions_2.mat','fake_running_speed_exp_stats_all_sessions');
    save('fake_running_speed_stats_all_sessions_2.mat','fake_running_speed_stats_all_sessions');
    
else
%     save('running_speed_stats_all_sessions_3.mat','running_speed_stats_all_sessions');
%     save('shuffled_running_speed_stats_all_sessions_3.mat','shuffled_running_speed_stats_all_sessions');
%     save('running_speed_exp_stats_all_sessions_3.mat','running_speed_exp_stats_all_sessions');
%     save('shuffled_running_speed_exp_stats_all_sessions_3.mat','shuffled_running_speed_exp_stats_all_sessions');
%     
%     save('left_turn_stats_all_sessions_3.mat','left_turn_stats_all_sessions');
%     save('shuffled_left_turn_stats_all_sessions_3.mat','shuffled_left_turn_stats_all_sessions');
%     save('left_turn_exp_stats_all_sessions_3.mat','left_turn_exp_stats_all_sessions');
%     save('shuffled_left_turn_exp_stats_all_sessions_3.mat','shuffled_left_turn_exp_stats_all_sessions');
%     
%     save('right_turn_stats_all_sessions_3.mat','right_turn_stats_all_sessions');
%     save('shuffled_right_turn_stats_all_sessions_3.mat','shuffled_right_turn_stats_all_sessions');
%     save('right_turn_exp_stats_all_sessions_3.mat','right_turn_exp_stats_all_sessions');
%     save('shuffled_right_turn_exp_stats_all_sessions_3.mat','shuffled_right_turn_exp_stats_all_sessions');
%     
%     save('fake_left_turn_stats_all_sessions_3.mat','fake_left_turn_stats_all_sessions');
%     save('fake_left_turn_exp_stats_all_sessions_3.mat','fake_left_turn_exp_stats_all_sessions');
%     save('fake_right_turn_stats_all_sessions_3.mat','fake_right_turn_stats_all_sessions');
%     save('fake_right_turn_exp_stats_all_sessions_3.mat','fake_right_turn_exp_stats_all_sessions');
%     
%     save('avg_max_rate_3.mat','avg_max_rate');
%     

 save('fake_running_speed_exp_stats_all_sessions_3.mat','fake_running_speed_exp_stats_all_sessions');
 save('fake_running_speed_stats_all_sessions_3.mat','fake_running_speed_stats_all_sessions');
end


% if strcmpi(mode,'first')
%     save('RSC_ISI_half_peaks_1.mat','RSC_ISI_half_peaks');
%     save('RSC_ISI_peaks_1.mat','RSC_ISI_peaks');
%     save('RSC_ISI_constant_a_1.mat','RSC_ISI_constant_a');
%     save('RSC_ISI_constant_b_1.mat','RSC_ISI_constant_b');
%     save('RSC_ISIs_1.mat','RSC_ISIs');
%
%     save('POS_ISI_half_peaks_1.mat','POS_ISI_half_peaks');
%     save('POS_ISI_peaks_1.mat','POS_ISI_peaks');
%     save('POS_ISI_constant_a_1.mat','POS_ISI_constant_a');
%     save('POS_ISI_constant_b_1.mat','POS_ISI_constant_b');
%     save('POS_ISIs_1.mat','POS_ISIs');
%
%     fileID = fopen('RSC_data_ordering_first.txt','w');
%     formatSpec = '%s\n';
%     for idx = 1:num_trials_this_session
%         fprintf(fileID,formatSpec,trial_labels{idx,:});
%     end
%     fclose(fileID);
% elseif strcmpi(mode,'second')
%     save('RSC_ISI_half_peaks2.mat','RSC_ISI_half_peaks');
%     save('RSC_ISI_peaks_2.mat','RSC_ISI_peaks');
%     save('RSC_ISI_constant_a_2.mat','RSC_ISI_constant_a');
%     save('RSC_ISI_constant_b_2.mat','RSC_ISI_constant_b');
%     save('RSC_ISIs_2.mat','RSC_ISIs');
%
%     save('POS_ISI_half_peaks_2.mat','POS_ISI_half_peaks');
%     save('POS_ISI_peaks_2.mat','POS_ISI_peaks');
%     save('POS_ISI_constant_a_2.mat','POS_ISI_constant_a');
%     save('POS_ISI_constant_b_2.mat','POS_ISI_constant_b');
%     save('POS_ISIs_2.mat','POS_ISIs');
%
%     fileID = fopen('RSC_data_ordering_second.txt','w');
%     formatSpec = '%s\n';
%     for idx = 1:num_trials_this_session
%         fprintf(fileID,formatSpec,trial_labels{idx,:});
%     end
%     fclose(fileID);
% else
%     save('RSC_ISI_half_peaks3.mat','RSC_ISI_half_peaks');
%     save('RSC_ISI_peaks_3.mat','RSC_ISI_peaks');
%     save('RSC_ISI_constant_a_3.mat','RSC_ISI_constant_a');
%     save('RSC_ISI_constant_b_3.mat','RSC_ISI_constant_b');
%     save('RSC_ISIs_3.mat','RSC_ISIs');
%
%     save('POS_ISI_half_peaks_3.mat','POS_ISI_half_peaks');
%     save('POS_ISI_peaks_3.mat','POS_ISI_peaks');
%     save('POS_ISI_constant_a_3.mat','POS_ISI_constant_a');
%     save('POS_ISI_constant_b_3.mat','POS_ISI_constant_b');
%     save('POS_ISIs_3.mat','POS_ISIs');
%
%     fileID = fopen('RSC_data_ordering_third.txt','w');
%     formatSpec = '%s\n';
%     for idx = 1:num_trials_this_session
%         fprintf(fileID,formatSpec,trial_labels{idx,:});
%     end
%     fclose(fileID);
% end
cd(startpath);
end

function out_array = nanshuffle(in_array)

%This function returns a version of in_array with all non-NaN elements
%shuffled. NaNs remain in the same place

temp = in_array(~isnan(in_array)); %all non-nan values of in_array
shuffled_temp = temp(randperm(length(temp))); %shuffled non-NaN elements only


out_array = zeros(1,numel(in_array)); %create one possible shuffled array
out_array(isnan(in_array)) = NaN; %put NaNs in the right place


out_array(~isnan(out_array)) = shuffled_temp;

end










