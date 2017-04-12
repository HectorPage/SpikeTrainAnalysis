function conjunctive_analysisV2(mode)
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


%% These are for the ATI calculations
ATIs = zeros(total_sessions,1);


%% Session Loop
for session = 1:total_sessions %loop over all the sessions
    
    
    disp('------------------------------------');
    disp(['Session:',num2str((session/total_sessions)*100),'%']);
    disp('------------------------------------');
    
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
    
    
    
    filepath = ['/Volumes/ibn/Jeffery lab/Hector Page/Yave data/Raw Dacq data/',areas{session},'/r',num2str(rats(session)),...
        '/',cards{session},'/',dates{session}]; %area rat card date
    cd(filepath);
    
    for trialnumber = 1:num_trials_this_session %do all the plotting for each trial of the data
        
        filename = ['r',num2str(rats(session)),'_',dates{session},'_',trial_labels{trialnumber}];
        full_filename = strcat(filename,'.set');
        %% READING DATA
        
        disp('------------------------------------');
        fprintf(['Fetching DACQ data for trial ',num2str(trialnumber),' of ',num2str(num_trials_this_session)]);
        mtint = readAllDACQdata(filepath,full_filename);
        disp(' done');
        
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
        
        Spks = mtint.tetrode(tetrodes(session)).cut==clusters(session);                                         %logical index of which spikes on this tetrode are clusters(session)
        SpikeTimes= mtint.tetrode(tetrodes(session)).ts(Spks);                                       %indexing in to extract spiketimes for this cell
        
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
        
        %% Getting preliminary data read in
        
        %HD bins - 6 deg width
        HD_binedges = 0:6:360;
        num_HD_bins = numel(HD_binedges)-1;
        
        %get HD read in
        Vars=daVarsStruct;
        in_polar1 = make_in_struct_for_rm (mtint,tetrodes(session),clusters(session),50,Vars.rm.smthKernPos,Vars.rm.binSizePosCm,Vars.rm.binSizeDir,0,Vars,'dir');
        if size(in_polar1.posDir,2) ~= 1
            in_polar1.posDir = in_polar1.posDir';
        end
        in = in_polar1;
        HeadDirection = in.posDir; %HD at 50ms sample rate
        
        %Speed
        speed = mtint.pos.speed; %this is presumably speed in cm/s
        
        %% PFD CHANGES ACROSS TRIALS OF SAME CELL, SO NEED TO CORRECT HD ON THIS BASIS
        %Get PFD of the HD cell this trial
        PolarMap_cell1 = GetRateMap (in); %using UCL legacy function to get polar rate map of this cell
        [~,sortIndex] = sort(PolarMap_cell1.map(:),'descend');
        HD_bin_centers = 3:6:357;
        PFD = HD_bin_centers(sortIndex(1)); %this is taken as PFD of the cell - bit inaccurate but will do
        
        fprintf(['PFD for trial ',num2str(trialnumber),' is ',num2str(PFD),' degrees']);
        
        HeadDirection = HeadDirection - PFD; %HeadDirection is now in terms of distance from PFD
        HeadDirection(HeadDirection<=0) = HeadDirection(HeadDirection<=0) + 360.0; %Make sure no negative HDs
        
        HeadDirection = HeadDirection - 180; %shifting by 180 just to get tuning curve in middle of plot - change x axis labels when plotting!
        HeadDirection(HeadDirection<=0) = HeadDirection(HeadDirection<=0) + 360.0; %Make sure no negative HDs
        
        
        %% Recording conglomerated spike times, (corrected) Head Directions, and AHVs - for analysis by session
        
        if trialnumber<2
            HD_all_trials = HeadDirection(:);
            SpikeTimes_all_trials = SpikeTimes(:);
            speed_all_trials = speed(:);
        else
            HD_all_trials = [HD_all_trials; HeadDirection(:)];
            SpikeTimes_all_trials = [SpikeTimes_all_trials;(SpikeTimes(:)+prev_tmax)];
            speed_all_trials = [speed_all_trials;speed(:)];
        end
        
        if trialnumber<2
            prev_tmax = tmax;
        else
            prev_tmax = prev_tmax+tmax;
        end
        
        %% Working out conjunctive HD by running speed tuning for this trial
        
        %Do the adaptive binning of running speed
        
        num_speed_bins = 20;
        y =  quantile(speed(speed>2),num_speed_bins-1);
        [counts, Id] = histc(speed(speed>2),[-inf; y(:); inf]); %tester line to see if bin counts in speed dimension are equal (i.e. equal dwell)
        speed_binedges = [2;y(:);inf];
        speed_vals = speed_binedges(1:end-1);
        for idx = 1:numel(speed_vals)-1 %last speedval is just the binedges as upper edge is infinity
            speed_vals(idx) = (mean(speed_vals(idx:idx+1)));
        end
        
        %get fresh dwell and spikes for each trial, overall session binning
        %etc now done at the end of trial loop
        RunningSpeedbyHDSpikes = zeros(num_speed_bins, num_HD_bins); %array for the spikecounts
        RunningSpeedbyHDDwell = zeros(num_speed_bins, num_HD_bins); %array for dwelltime
        
        %Now count trial dwells and spikes
        
        for speedindex = 1:num_speed_bins
            
            for HDindex = 1:num_HD_bins
                
                %find time bins where both variables are within bin range
                conjunctive_timebins = find(speed>=speed_binedges(speedindex) & speed<speed_binedges(speedindex+1)...
                    & HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1));
                
                RunningSpeedbyHDDwell(speedindex, HDindex) = RunningSpeedbyHDDwell(speedindex, HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
                
                for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                    lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                    upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                    
                    RunningSpeedbyHDSpikes(speedindex, HDindex) = RunningSpeedbyHDSpikes(speedindex, HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
                end
                
            end
        end
        
        %% Generating running speed by HD plot for this trial
        
        figure('units','normalized','outerposition',[0 0 1 1]);
        subplot(2,3,2:3);
        smoothing_kernel = 5;
        new_RunningSpeedbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,RunningSpeedbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
        new_RunningSpeedbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,RunningSpeedbyHDSpikes);
        
        %         new_RunningSpeedbyHDDwell = RunningSpeedbyHDDwell; %no longer smoothing
        %         new_RunningSpeedbyHDSpikes = RunningSpeedbyHDSpikes;
        
        RunningSpeedbyHDRates = new_RunningSpeedbyHDSpikes./new_RunningSpeedbyHDDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_RunningSpeedbyHDDwell(:))/ numel(new_RunningSpeedbyHDDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_RunningSpeedbyHDDwell<bin_exclusion_criterion;
        RunningSpeedbyHDRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(speed_vals,3:6:357);
        surf(X,Y,RunningSpeedbyHDRates');
        shading interp;
        grid off;
        cbar=([0 max(RunningSpeedbyHDRates(:))]);
        caxis(cbar);
        colormap(jet);
        ylim([0,360]);
        xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        title([' Max FR = ',num2str(max(RunningSpeedbyHDRates(:))),' Hz',' Trial: ',trial_labels{trialnumber}]);
        set(gca,'Fontsize',12);
        
        
        Run_Spikes = sum(RunningSpeedbyHDSpikes,2);
        Run_Dwell = sum(RunningSpeedbyHDDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        Run_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Dwell); %this is the smoothing from GetRateMap function in UCL store
        %No longer filtering Dwell!
        
        Run_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Spikes);
        
        Run_Rates = Run_Spikes./Run_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(Run_Dwell(:))/ numel(Run_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = Run_Dwell<bin_exclusion_criterion;
        Run_Rates(bin_exclusion_filter) = NaN;
        
        subplot(2,3,5:6);
        %Adding shuffle for running speed
        shuffled_run_rates = zeros(1000,numel(Run_Rates));
        
        for shuffle = 1:1000
            shuffled_RunbyHDSpikes = shuffle_conjunctive(RunningSpeedbyHDSpikes);
            
            Running_Speed_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_RunbyHDSpikes,2));
            %Running_Speed_Spikes = sum(shuffled_RunbyHDSpikes,2);
            
            Running_Speed_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(RunningSpeedbyHDDwell,2));
            
            %No longer filtering Dwell!
            %Running_Speed_Dwell = Run_Dwell;
            
            RunSpeed_Rates = Running_Speed_Spikes./Running_Speed_Dwell;
            RunSpeed_Rates(bin_exclusion_filter) = NaN;
            shuffled_run_rates(shuffle,:) = RunSpeed_Rates;
        end
        
        hold on
        boundedline(speed_vals,nanmean(shuffled_run_rates,1),nanstd(shuffled_run_rates,1).*2);
        plot(speed_vals,Run_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        ylabel('Firing Rate (Hz)');
        xlabel('Speed (cm/s)');
        ylim([0 Inf]);
        xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
        set(gca,'Fontsize',12);
        
        %Plotting just HD
        
        HD_Spk = sum(RunningSpeedbyHDSpikes,1);
        HD_Dwl = sum(RunningSpeedbyHDDwell,1);
        
        HD_Dwl = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Dwl); %this is the smoothing from GetRateMap function in UCL store
        HD_Spk = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Spk);
        
        HD_Rts = HD_Spk./HD_Dwl;
        
        bin_exclusion_criterion = 0.01 * (sum(HD_Dwl(:))/ numel(HD_Dwl(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = HD_Dwl<bin_exclusion_criterion;
        HD_Rts(bin_exclusion_filter) = NaN;
        
        h = subplot(2,3,1); %HD
        plot(3:6:357,HD_Rts, 'b', 'Linewidth', 2.0);
        ylabel('Firing Rate (Hz)');
        xlabel('HD - PFD (deg)');
        xlim([0 360]);
        set(gca,'Xtick',0:90:360);
        set(gca,'Xticklabel',-180:90:180);
        view([-90 90]); %orient correctly for subplotting
        set(gca,'Fontsize',12);
        cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/conjunctive_plots/per_trial_plots'); %save in the plots directory
        new_session_text = strrep(session_text{session},'.','');
        new_session_text = strrep(new_session_text,'\','_');
        saveas(gcf,[ new_session_text,'_conj_running_speed_',trial_labels{trialnumber}],'tif');
        close(gcf);
        
        
        %% Getting AHV values
        AngHeadVel = zeros(numel(HeadDirection),1);
        
        
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
        
        if trialnumber<2
            AngHeadVel_all_trials = AngHeadVel(:);
        else
            AngHeadVel_all_trials = [AngHeadVel_all_trials; AngHeadVel(:)];
        end
        
        
        %N.B. I have made the decision to exclude AHV = 0 data points, as
        %they are not actually head turns, and we are not sure what they
        %involve the rat doing...
        R_AHV = AngHeadVel(AngHeadVel>0);
        L_AHV = AngHeadVel(AngHeadVel<0);
        
        
        %Do the adaptive binning of AHV - do L and R separately
        
        num_AHV_bins = 20;
        
        y =  quantile(R_AHV,num_AHV_bins-1);
        [counts, Id] = histc(R_AHV,[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
        R_AHV_binedges = [0;y(:);inf];
        R_AHV_vals = R_AHV_binedges(1:end-1);
        for idx = 1:numel(R_AHV_vals)-1 %ignore last, as it is infinity
            R_AHV_vals(idx) = mean(R_AHV_vals(idx:idx+1));
        end
        
        y =  quantile(abs(L_AHV),num_AHV_bins-1);
        [counts, Id] = histc(abs(L_AHV),[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
        L_AHV_binedges = [0;y(:);inf];
        L_AHV_vals = L_AHV_binedges(1:end-1);
        for idx = 1:numel(L_AHV_vals)-1 %ignore last, as it is infinity
            L_AHV_vals(idx) = mean(L_AHV_vals(idx:idx+1));
        end
        %L AHV is all negative so make the vals etc negative
        L_AHV_binedges = flipud(-L_AHV_binedges);
        L_AHV_vals = flipud(-L_AHV_vals);
        
        
        
        %% Dealing with R AHV
        
        %Get a fresh one for each trial
        R_AHVbyHDSpikes = zeros(num_AHV_bins, num_HD_bins); %array for the spikecounts
        R_AHVbyHDDwell = zeros(num_AHV_bins, num_HD_bins); %array for dwelltime
        
        for AHVindex = 1:num_AHV_bins
            
            for HDindex = 1:num_HD_bins
                
                %find time bins where both variables are within bin range
                conjunctive_timebins = find(AngHeadVel>=R_AHV_binedges(AHVindex) & AngHeadVel<R_AHV_binedges(AHVindex+1)...
                    & HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1) & AngHeadVel~=0);
                
                R_AHVbyHDDwell(AHVindex, HDindex) = R_AHVbyHDDwell(AHVindex, HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
                
                for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                    lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                    upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                    
                    R_AHVbyHDSpikes(AHVindex, HDindex) = R_AHVbyHDSpikes(AHVindex, HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
                end
                
            end
        end
        
        %Plottng R AHV stuff
        
        figure('units','normalized','outerposition',[0 0 1 1]);
        rh = subplot(3,6,[4,5,10,11]);
        smoothing_kernel = 5;
        new_R_AHVbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
        new_R_AHVbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyHDSpikes);
        
        %         new_R_AHVbyHDDwell = R_AHVbyHDDwell; %no longer smoothing
        %         new_R_AHVbyHDSpikes = R_AHVbyHDSpikes;
        
        
        R_AHVbyHDRates = new_R_AHVbyHDSpikes./new_R_AHVbyHDDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_R_AHVbyHDDwell(:))/ numel(new_R_AHVbyHDDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_R_AHVbyHDDwell<bin_exclusion_criterion;
        R_AHVbyHDRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(R_AHV_vals,3:6:357);
        surf(X,Y,R_AHVbyHDRates');
        shading interp;
        grid off;
        max_right_heat_plot = max(R_AHVbyHDRates(:));
        cbar = [0,max_right_heat_plot];
        caxis(cbar);
        colormap(jet);
        ylim([0,360]);
        xlim([R_AHV_binedges(1),ceil(R_AHV_binedges(end-1))]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        title(['Trial: ',trial_labels{trialnumber}]);
        set(gca,'Fontsize',12);
        set(gca,'Yaxislocation','right');
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        
        R_AHV_Spikes = sum(R_AHVbyHDSpikes,2);
        R_AHV_Dwell = sum(R_AHVbyHDDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        R_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
        
        %No longer filtering Dwell!
        R_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Spikes);
        
        R_AHV_Rates = R_AHV_Spikes./R_AHV_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(R_AHV_Dwell(:))/ numel(R_AHV_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = R_AHV_Dwell<bin_exclusion_criterion & R_AHV_Spikes>2;
        R_AHV_Rates(bin_exclusion_filter) = NaN;
        
        rahv = subplot(3,6,[16,17]);
        %Adding shuffle for running speed
        shuffled_R_AHV_rates = zeros(1000,numel(R_AHV_Rates));
        
        for shuffle = 1:1000
            shuffled_R_AHV_byHDSpikes = shuffle_conjunctive(R_AHVbyHDSpikes);
            
            R_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_R_AHV_byHDSpikes,2));
            %R_AHV_Spikes =  sum(shuffled_R_AHV_byHDSpikes,2);
            R_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(R_AHVbyHDDwell,2));
            %No longer filtering Dwell!
            
            
            R_AHV_Rates_shuff = R_AHV_Spikes./R_AHV_Dwell;
            R_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
            shuffled_R_AHV_rates(shuffle,:) = R_AHV_Rates_shuff;
        end
        
        hold on
        boundedline(R_AHV_vals,nanmean(shuffled_R_AHV_rates,1),nanstd(shuffled_R_AHV_rates,1).*2);
        plot(R_AHV_vals,R_AHV_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        
        xlabel('R AHV (deg/s)');
        r_max_ahv_fr = max(R_AHV_Rates);
        ylim([0 Inf]);
        xlim([0,ceil(R_AHV_binedges(end-1))]);
        set(gca,'Fontsize',12);
        set(gca,'YColor','none');
        set(gca,'YTicklabel',[]);
        
        %Plotting just HD
        
        HD_Spk = sum(R_AHVbyHDSpikes,1);
        HD_Dwl = sum(R_AHVbyHDDwell,1);
        
        HD_Dwl = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Dwl); %this is the smoothing from GetRateMap function in UCL store
        HD_Spk = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Spk);
        
        HD_Rts = HD_Spk./HD_Dwl;
        
        bin_exclusion_criterion = 0.01 * (sum(HD_Dwl(:))/ numel(HD_Dwl(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = HD_Dwl<bin_exclusion_criterion;
        HD_Rts(bin_exclusion_filter) = NaN;
        
        rhd = subplot(3,6,[6,12]);
        plot(-177:6:177,HD_Rts, 'b', 'Linewidth', 2.0);
        ylabel('Firing Rate (Hz)');
        xlabel('HD - PFD (deg)');
        ylim([0 Inf]);
        r_max_hd_fr = max(HD_Rts);
        xlim([-180 180]);
        set(gca,'XTick',-180:90:180);
        set(gca,'Xaxislocation','top');
        view([90 90]); %orient correctly for subplotting
        set(gca,'XDir','reverse');
        set(gca,'YDir','reverse');
        box off;
        set(gca,'Fontsize',12);
       
        
        %% Now Dealing with L AHV
        
        %Get a fresh one for each trial
        L_AHVbyHDSpikes = zeros(num_AHV_bins, num_HD_bins); %array for the spikecounts
        L_AHVbyHDDwell = zeros(num_AHV_bins, num_HD_bins); %array for dwelltime
        
        for AHVindex = 1:num_AHV_bins
            
            for HDindex = 1:num_HD_bins
                
                %find time bins where both variables are within bin range
                conjunctive_timebins = find(AngHeadVel>=L_AHV_binedges(AHVindex) & AngHeadVel<L_AHV_binedges(AHVindex+1)...
                    & HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1) & AngHeadVel~=0);
                
                L_AHVbyHDDwell(AHVindex, HDindex) = L_AHVbyHDDwell(AHVindex, HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
                
                for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                    lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                    upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                    
                    L_AHVbyHDSpikes(AHVindex, HDindex) = L_AHVbyHDSpikes(AHVindex, HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
                end
                
            end
        end
        
        %Plottng L AHV stuff
        
        subplot(3,6,[2,3,8,9]);
        smoothing_kernel = 5;
        new_L_AHVbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
        new_L_AHVbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyHDSpikes);
        
        %         new_L_AHVbyHDDwell = L_AHVbyHDDwell; %no longer smoothing
        %         new_L_AHVbyHDSpikes = L_AHVbyHDSpikes;
        
        L_AHVbyHDRates = new_L_AHVbyHDSpikes./new_L_AHVbyHDDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_L_AHVbyHDDwell(:))/ numel(new_L_AHVbyHDDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_L_AHVbyHDDwell<bin_exclusion_criterion;
        L_AHVbyHDRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(L_AHV_vals,3:6:357);
        surf(X,Y,L_AHVbyHDRates');
        shading interp;
        grid off;
         if max(L_AHVbyHDRates(:)) > max_right_heat_plot
            caxis(rh,[0,max(L_AHVbyHDRates(:))]); % set the clim on right heat map
            caxis([0,max(L_AHVbyHDRates(:))]); % set the clim on this heat map
            title(['Max FR = ',num2str(max(L_AHVbyHDRates(:))),' Hz']);
        else
            caxis(cbar); %use cbar from right heat map
            title(['Max FR = ',num2str(max_right_heat_plot),' Hz']);
        end
        colormap(jet);
        ylim([0,360]);
        xlim([ceil(L_AHV_binedges(2)),L_AHV_binedges(end)]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        set(gca,'Fontsize',12);
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        
        L_AHV_Spikes = sum(L_AHVbyHDSpikes,2);
        L_AHV_Dwell = sum(L_AHVbyHDDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        L_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
        %No longer filtering Dwell!
        
        L_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Spikes);
        
        L_AHV_Rates = L_AHV_Spikes./L_AHV_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(L_AHV_Dwell(:))/ numel(L_AHV_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = L_AHV_Dwell<bin_exclusion_criterion & L_AHV_Spikes>2;
        L_AHV_Rates(bin_exclusion_filter) = NaN;
        
        subplot(3,6,[14,15]);
        %Adding shuffle for AHV
        shuffled_L_AHV_rates = zeros(1000,numel(L_AHV_Rates));
        
        for shuffle = 1:1000
            shuffled_L_AHV_byHDSpikes = shuffle_conjunctive(L_AHVbyHDSpikes);
            
            L_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_L_AHV_byHDSpikes,2));
            %L_AHV_Spikes =  sum(shuffled_L_AHV_byHDSpikes,2);
            L_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(L_AHVbyHDDwell,2));
            %No longer filtering Dwell!
            
            L_AHV_Rates_shuff = L_AHV_Spikes./L_AHV_Dwell;
            L_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
            shuffled_L_AHV_rates(shuffle,:) = L_AHV_Rates_shuff;
        end
        
        hold on
        boundedline(L_AHV_vals,nanmean(shuffled_L_AHV_rates,1),nanstd(shuffled_L_AHV_rates,1).*2);
        plot(L_AHV_vals,L_AHV_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        ylabel('Firing Rate (Hz)');
        xlabel('L AHV (deg/s)');
         
        if max(L_AHV_Rates) > r_max_ahv_fr
            ylim([0, max(L_AHV_Rates)]);
            set(rahv,'ylim',[0, max(L_AHV_Rates)]);
        else
            ylim([0 r_max_ahv_fr]);
        end
        
        xlim([ceil(L_AHV_binedges(2)),0]);
        set(gca,'Fontsize',12);
        
        %Plotting just HD
        
        HD_Spk = sum(L_AHVbyHDSpikes,1);
        HD_Dwl = sum(L_AHVbyHDDwell,1);
        
        HD_Dwl = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Dwl); %this is the smoothing from GetRateMap function in UCL store
        HD_Spk = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Spk);
        
        HD_Rts = HD_Spk./HD_Dwl;
        
        bin_exclusion_criterion = 0.01 * (sum(HD_Dwl(:))/ numel(HD_Dwl(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = HD_Dwl<bin_exclusion_criterion;
        HD_Rts(bin_exclusion_filter) = NaN;
        
        subplot(3,6,[1,7]);
        plot(3:6:357,HD_Rts, 'b', 'Linewidth', 2.0);
        ylabel('Firing Rate (Hz)');
        xlim([0 360]);
        set(gca,'Xticklabel',[]);
        
         if max(HD_Rts) > r_max_hd_fr
            ylim([0, max(HD_Rts)]);
            set(rhd,'ylim',[0, max(HD_Rts)]);
        else
            ylim([0, r_max_hd_fr]);
         end
        
        set(gca,'Fontsize',12);
         view([90 90]); %orient correctly for subplotting
        set(gca,'XDir','reverse');
        set(gca,'YDir','reverse');
        set(gca,'XColor','none');
        box off;
        
        cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/conjunctive_plots/per_trial_plots'); %save in the plots directory
        new_session_text = strrep(session_text{session},'.','');
        new_session_text = strrep(new_session_text,'\','_');
        saveas(gcf,[ new_session_text,'_conj_AHV_',trial_labels{trialnumber}],'tif');
        close(gcf);
        
        
        
        %% Now doing running speed by AHV
        
        %Equalising dwell times here needs to be different for these conjunctive plots, because second
        %variable, given limited range, affects binning of first variable (unlike with HD).
        
        %Redoing binning of R_AHV and Running speed here based on
        %conjunctive limits......
        
        R_AHV = AngHeadVel(AngHeadVel>0 & speed>2);
        y =  quantile(R_AHV,num_AHV_bins-1);
        [counts, Id] = histc(R_AHV,[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
        R_AHV_binedges = [0;y(:);inf];
        R_AHV_vals = R_AHV_binedges(1:end-1);
        for idx = 1:numel(R_AHV_vals)-1 %ignore last, as it is infinity
            R_AHV_vals(idx) = mean(R_AHV_vals(idx:idx+1));
        end
        
        num_speed_bins = 20;
        y =  quantile(speed(speed>2 & AngHeadVel>0),num_speed_bins-1);
        [counts, Id] = histc(speed(speed>2),[-inf; y(:); inf]); %tester line to see if bin counts in speed dimension are equal (i.e. equal dwell)
        speed_binedges = [2;y(:);inf];
        speed_vals = speed_binedges(1:end-1);
        for idx = 1:numel(speed_vals)-1 %last speedval is just the binedges as upper edge is infinity
            speed_vals(idx) = (mean(speed_vals(idx:idx+1)));
        end
        
        
        %Get a fresh one for each trial
        R_AHVbyRunningSpeedSpikes = zeros(num_AHV_bins, num_speed_bins); %array for the spikecounts
        R_AHVbyRunningSpeedDwell = zeros(num_AHV_bins, num_speed_bins); %array for dwelltime
        
        for AHVindex = 1:num_AHV_bins
            
            for Speedindex = 1:num_speed_bins
                
                %find time bins where both variables are within bin range
                conjunctive_timebins = find(AngHeadVel>=R_AHV_binedges(AHVindex) & AngHeadVel<R_AHV_binedges(AHVindex+1)...
                    & speed>=speed_binedges(Speedindex) & speed<speed_binedges(Speedindex+1) & AngHeadVel~=0);
                
                R_AHVbyRunningSpeedDwell(AHVindex, Speedindex) = R_AHVbyRunningSpeedDwell(AHVindex, Speedindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
                
                for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                    lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                    upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                    
                    R_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) = R_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
                end
                
            end
        end
        
        %Plottng R AHV stuff
        
        
        figure('units','normalized','outerposition',[0 0 1 1]);
        rh = subplot(3,6,[4,5,10,11]);
        smoothing_kernel = 5;
        new_R_AHVbyRunningSpeedDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyRunningSpeedDwell); %this is the smoothing from GetRateMap function in UCL store
        new_R_AHVbyRunningSpeedSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyRunningSpeedSpikes);
        
        %         new_R_AHVbyRunningSpeedDwell = R_AHVbyRunningSpeedDwell; %no longer smoothing
        %         new_R_AHVbyRunningSpeedSpikes = R_AHVbyRunningSpeedSpikes;
        
        
        R_AHVbyRunningSpeedRates = new_R_AHVbyRunningSpeedSpikes./new_R_AHVbyRunningSpeedDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_R_AHVbyRunningSpeedDwell(:))/ numel(new_R_AHVbyRunningSpeedDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_R_AHVbyRunningSpeedDwell<bin_exclusion_criterion;
        R_AHVbyRunningSpeedRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(R_AHV_vals,speed_vals);
        surf(X,Y,R_AHVbyRunningSpeedRates');
        shading interp;
        grid off;
        max_right_heat_plot = max(R_AHVbyRunningSpeedRates(:));
        top_speed_right = max(speed_vals);
        cbar = [0,max_right_heat_plot];
        caxis(cbar);
        colormap(jet);
        ylim([speed_binedges(1),ceil(speed_binedges(end-1))]);
        xlim([R_AHV_binedges(1),ceil(R_AHV_binedges(end-1))]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        title(['Trial: ',trial_labels{trialnumber}]);
        set(gca,'Fontsize',12);
        set(gca,'Yaxislocation','right');
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        
        R_AHV_Spikes = sum(R_AHVbyRunningSpeedSpikes,2);
        R_AHV_Dwell = sum(R_AHVbyRunningSpeedDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        R_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
        
        %No longer filtering Dwell!
        R_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Spikes);
        
        R_AHV_Rates = R_AHV_Spikes./R_AHV_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(R_AHV_Dwell(:))/ numel(R_AHV_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = R_AHV_Dwell<bin_exclusion_criterion & R_AHV_Spikes>2;
        R_AHV_Rates(bin_exclusion_filter) = NaN;
        
        rahv = subplot(3,6,[16,17]);
        %Adding shuffle for running speed
        shuffled_R_AHV_rates = zeros(1000,numel(R_AHV_Rates));
        
        for shuffle = 1:1000
            shuffled_R_AHV_byRunningSpeedSpikes = shuffle_conjunctive(R_AHVbyRunningSpeedSpikes);
            
            R_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_R_AHV_byRunningSpeedSpikes,2));
            %R_AHV_Spikes =  sum(shuffled_R_AHV_byHDSpikes,2);
            R_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(R_AHVbyRunningSpeedDwell,2));
            %No longer filtering Dwell!
            
            
            R_AHV_Rates_shuff = R_AHV_Spikes./R_AHV_Dwell;
            R_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
            shuffled_R_AHV_rates(shuffle,:) = R_AHV_Rates_shuff;
        end
        
        hold on
        boundedline(R_AHV_vals,nanmean(shuffled_R_AHV_rates,1),nanstd(shuffled_R_AHV_rates,1).*2);
        plot(R_AHV_vals,R_AHV_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        
        xlabel('R AHV (deg/s)');
        ylim([0 Inf]);
        r_max_ahv_fr = max(R_AHV_Rates);
        xlim([0,ceil(R_AHV_binedges(end-1))]);
        set(gca,'Fontsize',12);
        set(gca,'YColor','none');
        set(gca,'YTicklabel',[]);
        
        %Plotting just Running Speed
        
        Run_Spikes = sum(R_AHVbyRunningSpeedSpikes,1);
        Run_Dwell = sum(R_AHVbyRunningSpeedDwell,1);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        Run_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Dwell); %this is the smoothing from GetRateMap function in UCL store
        %No longer filtering Dwell!
        
        Run_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Spikes);
        
        Run_Rates = Run_Spikes./Run_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(Run_Dwell(:))/ numel(Run_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = Run_Dwell<bin_exclusion_criterion;
        Run_Rates(bin_exclusion_filter) = NaN;
        
        rs = subplot(3,6,[6,12]);
        %Adding shuffle for running speed
        shuffled_run_rates = zeros(1000,numel(Run_Rates));
        
        for shuffle = 1:1000
            shuffled_RunbyHDSpikes = shuffle_conjunctive(RunningSpeedbyHDSpikes);
            
            Running_Speed_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_RunbyHDSpikes,2));
            %Running_Speed_Spikes = sum(shuffled_RunbyHDSpikes,2);
            
            Running_Speed_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(RunningSpeedbyHDDwell,2));
            
            %No longer filtering Dwell!
            %Running_Speed_Dwell = Run_Dwell;
            
            RunSpeed_Rates = Running_Speed_Spikes./Running_Speed_Dwell;
            RunSpeed_Rates(bin_exclusion_filter) = NaN;
            shuffled_run_rates(shuffle,:) = RunSpeed_Rates;
        end
        
        hold on
        boundedline(speed_vals,nanmean(shuffled_run_rates,1),nanstd(shuffled_run_rates,1).*2);
        plot(speed_vals,Run_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        ylabel('Firing Rate (Hz)');
        xlabel('Speed (cm/s)');
        ylim([0 Inf]);
        r_max_speed_fr = max(Run_Rates);
        xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
        set(gca,'Xaxislocation','top');
        view([90 90]); %orient correctly for subplotting
        set(gca,'XDir','reverse');
        set(gca,'YDir','reverse');
        set(gca,'Fontsize',12);
        
        %% Now doing running speed by L AHV
        %Equalising dwell times here needs to be different for these conjunctive plots, because second
        %variable, given limited range, affects binning of first variable (unlike with HD).
        
        %Redoing binning of R_AHV and Running speed here based on
        %conjunctive limits......
        
        L_AHV = AngHeadVel(AngHeadVel<0 & speed>2);
        y =  quantile(abs(L_AHV),num_AHV_bins-1);
        [counts, Id] = histc(abs(L_AHV),[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
        L_AHV_binedges = [0;y(:);inf];
        L_AHV_vals = L_AHV_binedges(1:end-1);
        for idx = 1:numel(L_AHV_vals)-1 %ignore last, as it is infinity
            L_AHV_vals(idx) = mean(L_AHV_vals(idx:idx+1));
        end
        %L AHV is all negative so make the vals etc negative
        L_AHV_binedges = flipud(-L_AHV_binedges);
        L_AHV_vals = flipud(-L_AHV_vals);
        
        num_speed_bins = 20;
        y =  quantile(speed(speed>2 & AngHeadVel<0),num_speed_bins-1);
        [counts, Id] = histc(speed(speed>2),[-inf; y(:); inf]); %tester line to see if bin counts in speed dimension are equal (i.e. equal dwell)
        speed_binedges = [2;y(:);inf];
        speed_vals = speed_binedges(1:end-1);
        for idx = 1:numel(speed_vals)-1 %last speedval is just the binedges as upper edge is infinity
            speed_vals(idx) = (mean(speed_vals(idx:idx+1)));
        end
        
        
        %Get a fresh one for each trial
        L_AHVbyRunningSpeedSpikes = zeros(num_AHV_bins, num_speed_bins); %array for the spikecounts
        L_AHVbyRunningSpeedDwell = zeros(num_AHV_bins, num_speed_bins); %array for dwelltime
        
        for AHVindex = 1:num_AHV_bins
            
            for Speedindex = 1:num_speed_bins
                
                %find time bins where both variables are within bin range
                conjunctive_timebins = find(AngHeadVel>=L_AHV_binedges(AHVindex) & AngHeadVel<L_AHV_binedges(AHVindex+1)...
                    & speed>=speed_binedges(Speedindex) & speed<speed_binedges(Speedindex+1) & AngHeadVel~=0);
                
                L_AHVbyRunningSpeedDwell(AHVindex, Speedindex) = L_AHVbyRunningSpeedDwell(AHVindex, Speedindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
                
                for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                    lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                    upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                    
                    L_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) = L_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
                end
                
            end
        end
        
        %Plottng L AHV stuff
        
        
        subplot(3,6,[2,3,8,9]);
        smoothing_kernel = 5;
        new_L_AHVbyRunningSpeedDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyRunningSpeedDwell); %this is the smoothing from GetRateMap function in UCL store
        new_L_AHVbyRunningSpeedSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyRunningSpeedSpikes);
        
        %         new_L_AHVbyRunningSpeedDwell = L_AHVbyRunningSpeedDwell; %no longer smoothing
        %         new_L_AHVbyRunningSpeedSpikes = L_AHVbyRunningSpeedSpikes;
        
        
        L_AHVbyRunningSpeedRates = new_L_AHVbyRunningSpeedSpikes./new_L_AHVbyRunningSpeedDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_L_AHVbyRunningSpeedDwell(:))/ numel(new_L_AHVbyRunningSpeedDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_L_AHVbyRunningSpeedDwell<bin_exclusion_criterion;
        L_AHVbyRunningSpeedRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(L_AHV_vals,speed_vals);
        surf(X,Y,L_AHVbyRunningSpeedRates');
        shading interp;
        grid off;
        %cbar=([0 max(L_AHVbyRunningSpeedRates(:))]);
        if max(L_AHVbyRunningSpeedRates(:)) > max_right_heat_plot
            caxis(rh,[0,max(L_AHVbyRunningSpeedRates(:))]); % set the clim on right heat map
            caxis([0,max(L_AHVbyRunningSpeedRates(:))]); % set the clim on this heat map
            title(['Max FR = ',num2str(max(L_AHVbyRunningSpeedRates(:))),' Hz']);
        else
            caxis(cbar); %use cbar from right heat map
            title(['Max FR = ',num2str(max_right_heat_plot),' Hz']);
        end
        
        colormap(jet);
        %Also need to do pratting around with heat maps y limit for running
        %speed....
        if speed_binedges(end-1) < top_speed_right
            ylim([speed_binedges(1),ceil(speed_binedges(end-1))]);
            set(rh,'ylim',[speed_binedges(1),ceil(speed_binedges(end-1))]);
        else
            ylim([speed_binedges(1),top_speed_right]);
        end
        xlim([ceil(L_AHV_binedges(2)),0]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        
        set(gca,'Fontsize',12);
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        
        L_AHV_Spikes = sum(L_AHVbyRunningSpeedSpikes,2);
        L_AHV_Dwell = sum(L_AHVbyRunningSpeedDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        L_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
        
        %No longer filtering Dwell!
        L_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Spikes);
        
        L_AHV_Rates = L_AHV_Spikes./L_AHV_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(L_AHV_Dwell(:))/ numel(L_AHV_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = L_AHV_Dwell<bin_exclusion_criterion & L_AHV_Spikes>2;
        L_AHV_Rates(bin_exclusion_filter) = NaN;
        
        subplot(3,6,[14,15]);
        %Adding shuffle for running speed
        shuffled_L_AHV_rates = zeros(1000,numel(L_AHV_Rates));
        
        for shuffle = 1:1000
            shuffled_L_AHV_byRunningSpeedSpikes = shuffle_conjunctive(L_AHVbyRunningSpeedSpikes);
            
            L_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_L_AHV_byRunningSpeedSpikes,2));
            %L_AHV_Spikes =  sum(shuffled_L_AHV_byHDSpikes,2);
            L_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(L_AHVbyRunningSpeedDwell,2));
            %No longer filtering Dwell!
            
            
            L_AHV_Rates_shuff = L_AHV_Spikes./L_AHV_Dwell;
            L_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
            shuffled_L_AHV_rates(shuffle,:) = L_AHV_Rates_shuff;
        end
        
        hold on
        boundedline(L_AHV_vals,nanmean(shuffled_L_AHV_rates,1),nanstd(shuffled_L_AHV_rates,1).*2);
        plot(L_AHV_vals,L_AHV_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        ylabel('Firing Rate (Hz)');
        xlabel('L AHV (deg/s)');
        
        if max(L_AHV_Rates) > r_max_ahv_fr
            ylim([0 max(L_AHV_Rates)]);
            set(rahv,'ylim',[0 max(L_AHV_Rates)]);
        else
            ylim([0 r_max_ahv_fr]);
        end
        xlim([ceil(L_AHV_binedges(2)),0]);
        set(gca,'Fontsize',12);
        
        %Plotting just Running Speed
        
        Run_Spikes = sum(L_AHVbyRunningSpeedSpikes,1);
        Run_Dwell = sum(L_AHVbyRunningSpeedDwell,1);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        Run_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Dwell); %this is the smoothing from GetRateMap function in UCL store
        %No longer filtering Dwell!
        
        Run_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Spikes);
        
        Run_Rates = Run_Spikes./Run_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(Run_Dwell(:))/ numel(Run_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = Run_Dwell<bin_exclusion_criterion;
        Run_Rates(bin_exclusion_filter) = NaN;
        
        subplot(3,6,[1,7]);
        %Adding shuffle for running speed
        shuffled_run_rates = zeros(1000,numel(Run_Rates));
        
        for shuffle = 1:1000
            shuffled_RunbyHDSpikes = shuffle_conjunctive(RunningSpeedbyHDSpikes);
            
            Running_Speed_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_RunbyHDSpikes,2));
            %Running_Speed_Spikes = sum(shuffled_RunbyHDSpikes,2);
            
            Running_Speed_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(RunningSpeedbyHDDwell,2));
            
            %No longer filtering Dwell!
            %Running_Speed_Dwell = Run_Dwell;
            
            RunSpeed_Rates = Running_Speed_Spikes./Running_Speed_Dwell;
            RunSpeed_Rates(bin_exclusion_filter) = NaN;
            shuffled_run_rates(shuffle,:) = RunSpeed_Rates;
        end
        
        hold on
        boundedline(speed_vals,nanmean(shuffled_run_rates,1),nanstd(shuffled_run_rates,1).*2);
        plot(speed_vals,Run_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        ylabel('Firing Rate (Hz)');
        %xlabel('Speed (cm/s)');
        set(gca,'Xticklabel',[]);
        ylim([0 Inf]);
        
        if max(Run_Rates) > r_max_speed_fr
            ylim([0, max(Run_Rates)]);
            set(rs,'ylim',[0, max(Run_Rates)]);
        else
            ylim([0, r_max_speed_fr]);
        end
        
        if speed_binedges(end-1) < top_speed_right
            xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
            set(rs,'ylim',[speed_binedges(1),ceil(speed_binedges(end-1))]);
        else
            xlim([speed_binedges(1),top_speed_right]);
        end
        
        set(gca,'Fontsize',12);
        view([90 90]); %orient correctly for subplotting
        set(gca,'XDir','reverse');
        set(gca,'YDir','reverse');
        set(gca,'XColor','none');
        
        cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/conjunctive_plots/per_trial_plots'); %save in the plots directory
        new_session_text = strrep(session_text{session},'.','');
        new_session_text = strrep(new_session_text,'\','_');
        saveas(gcf,[ new_session_text,'_conj_speed_AHV_',trial_labels{trialnumber}],'tif');
        close(gcf);
        fclose('all');
        
        cd(startpath); %GO HOME
        
    end
    
    cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/conjunctive_plots/per_session_plots'); %save in the plots directory
    
    %% NOW DOING ON PER SESSION BASIS
    
    HeadDirection = HD_all_trials;
    speed = speed_all_trials;
    SpikeTimes = SpikeTimes_all_trials;
    AngHeadVel = AngHeadVel_all_trials;
    [SpikeTrain,~] = histcounts(SpikeTimes,0:0.02:prev_tmax); %in case needed for Aman's analysis
    
    %% Running speed by HD
    
    %Do the adaptive binning of running speed
    
    num_speed_bins = 20;
    y =  quantile(speed(speed>2),num_speed_bins-1);
    [counts, Id] = histc(speed(speed>2),[-inf; y(:); inf]); %tester line to see if bin counts in speed dimension are equal (i.e. equal dwell)
    speed_binedges = [2;y(:);inf];
    speed_vals = speed_binedges(1:end-1);
    for idx = 1:numel(speed_vals)-1 %last speedval is just the binedges as upper edge is infinity
        speed_vals(idx) = (mean(speed_vals(idx:idx+1)));
    end
    
    %get fresh dwell and spikes for each trial, overall session binning
    %etc now done at the end of trial loop
    RunningSpeedbyHDSpikes = zeros(num_speed_bins, num_HD_bins); %array for the spikecounts
    RunningSpeedbyHDDwell = zeros(num_speed_bins, num_HD_bins); %array for dwelltime
    
    %Now count trial dwells and spikes
    
    for speedindex = 1:num_speed_bins
        
        for HDindex = 1:num_HD_bins
            
            %find time bins where both variables are within bin range
            conjunctive_timebins = find(speed>=speed_binedges(speedindex) & speed<speed_binedges(speedindex+1)...
                & HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1));
            
            RunningSpeedbyHDDwell(speedindex, HDindex) = RunningSpeedbyHDDwell(speedindex, HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
            
            for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                
                RunningSpeedbyHDSpikes(speedindex, HDindex) = RunningSpeedbyHDSpikes(speedindex, HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
            end
            
        end
    end
    
    
    
    %% Generating running speed by HD plot for this session
    
    figure('units','normalized','outerposition',[0 0 1 1]);
    subplot(2,3,2:3);
    smoothing_kernel = 5;
    new_RunningSpeedbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,RunningSpeedbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
    new_RunningSpeedbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,RunningSpeedbyHDSpikes);
    
    %         new_RunningSpeedbyHDDwell = RunningSpeedbyHDDwell; %no longer smoothing
    %         new_RunningSpeedbyHDSpikes = RunningSpeedbyHDSpikes;
    
    RunningSpeedbyHDRates = new_RunningSpeedbyHDSpikes./new_RunningSpeedbyHDDwell;
    
    bin_exclusion_criterion = 0.01 * (sum(new_RunningSpeedbyHDDwell(:))/ numel(new_RunningSpeedbyHDDwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = new_RunningSpeedbyHDDwell<bin_exclusion_criterion;
    RunningSpeedbyHDRates(bin_exclusion_filter) = NaN;
    
    [X,Y] = meshgrid(speed_vals,3:6:357);
    surf(X,Y,RunningSpeedbyHDRates');
    shading interp;
    grid off;
    cbar=([0 max(RunningSpeedbyHDRates(:))]);
    caxis(cbar);
    colormap(jet);
    ylim([0,360]);
    xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
    set(gca,'YTick',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'XTick',[]);
    set(gca,'XTickLabel',[]);
    view([0 90]);
    set(gca,'XColor','none');
    set(gca,'YColor','none');
    new_session_text = strrep(session_text{session},'.','');
    new_session_text = strrep(new_session_text,'\','_');
    title([new_session_text,' Max FR = ',num2str(max(RunningSpeedbyHDRates(:))),' Hz']);
    set(gca,'Fontsize',12);
    
    
    Run_Spikes = sum(RunningSpeedbyHDSpikes,2);
    Run_Dwell = sum(RunningSpeedbyHDDwell,2);
    
    %NOTE: this filter kernel really effects the edges of the Dwell
    %distribution - is this a problem?
    Run_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Dwell); %this is the smoothing from GetRateMap function in UCL store
    %No longer filtering Dwell!
    
    Run_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Spikes);
    
    Run_Rates = Run_Spikes./Run_Dwell;
    
    bin_exclusion_criterion = 0.01 * (sum(Run_Dwell(:))/ numel(Run_Dwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = Run_Dwell<bin_exclusion_criterion;
    Run_Rates(bin_exclusion_filter) = NaN;
    
    subplot(2,3,5:6);
    %Adding shuffle for running speed
    shuffled_run_rates = zeros(1000,numel(Run_Rates));
    
    for shuffle = 1:1000
        shuffled_RunbyHDSpikes = shuffle_conjunctive(RunningSpeedbyHDSpikes);
        
        Running_Speed_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_RunbyHDSpikes,2));
        %Running_Speed_Spikes = sum(shuffled_RunbyHDSpikes,2);
        
        Running_Speed_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(RunningSpeedbyHDDwell,2));
        
        %No longer filtering Dwell!
        %Running_Speed_Dwell = Run_Dwell;
        
        RunSpeed_Rates = Running_Speed_Spikes./Running_Speed_Dwell;
        RunSpeed_Rates(bin_exclusion_filter) = NaN;
        shuffled_run_rates(shuffle,:) = RunSpeed_Rates;
    end
    
    hold on
    boundedline(speed_vals,nanmean(shuffled_run_rates,1),nanstd(shuffled_run_rates,1).*2);
    plot(speed_vals,Run_Rates,'b', 'Linewidth', 2.0);
    hold off
    
    ylabel('Firing Rate (Hz)');
    xlabel('Speed (cm/s)');
    ylim([0 Inf]);
    xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
    set(gca,'Fontsize',12);
    
    %Plotting just HD
    
    HD_Spk = sum(RunningSpeedbyHDSpikes,1);
    HD_Dwl = sum(RunningSpeedbyHDDwell,1);
    
    HD_Dwl = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Dwl); %this is the smoothing from GetRateMap function in UCL store
    HD_Spk = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Spk);
    
    HD_Rts = HD_Spk./HD_Dwl;
    
    bin_exclusion_criterion = 0.01 * (sum(HD_Dwl(:))/ numel(HD_Dwl(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = HD_Dwl<bin_exclusion_criterion;
    HD_Rts(bin_exclusion_filter) = NaN;
    
    h = subplot(2,3,1); %HD
    plot(3:6:357,HD_Rts, 'b', 'Linewidth', 2.0);
    ylabel('Firing Rate (Hz)');
    xlabel('HD - PFD (deg)');
    xlim([0 360]);
    set(gca,'Xtick',0:90:360);
    set(gca,'Xticklabel',-180:90:180);
    view([-90 90]); %orient correctly for subplotting
    set(gca,'Fontsize',12);
    new_session_text = strrep(session_text{session},'.','');
    new_session_text = strrep(new_session_text,'\','_');
    saveas(gcf,[ new_session_text,'_conj_running_speed'],'tif');
    close(gcf);
    
    
    %% NOW AHV
    
    %N.B. I have made the decision to exclude AHV = 0 data points, as
    %they are not actually head turns, and we are not sure what they
    %involve the rat doing...
    R_AHV = AngHeadVel(AngHeadVel>0);
    L_AHV = AngHeadVel(AngHeadVel<0);
    
    
    %Do the adaptive binning of AHV - do L and R separately
    
    num_AHV_bins = 20;
    
    y =  quantile(R_AHV,num_AHV_bins-1);
    [counts, Id] = histc(R_AHV,[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
    R_AHV_binedges = [0;y(:);inf];
    R_AHV_vals = R_AHV_binedges(1:end-1);
    for idx = 1:numel(R_AHV_vals)-1 %ignore last, as it is infinity
        R_AHV_vals(idx) = mean(R_AHV_vals(idx:idx+1));
    end
    
    y =  quantile(abs(L_AHV),num_AHV_bins-1);
    [counts, Id] = histc(abs(L_AHV),[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
    L_AHV_binedges = [0;y(:);inf];
    L_AHV_vals = L_AHV_binedges(1:end-1);
    for idx = 1:numel(L_AHV_vals)-1 %ignore last, as it is infinity
        L_AHV_vals(idx) = mean(L_AHV_vals(idx:idx+1));
    end
    %L AHV is all negative so make the vals etc negative
    L_AHV_binedges = flipud(-L_AHV_binedges);
    L_AHV_vals = flipud(-L_AHV_vals);
    
    
    
    %% Dealing with R AHV
    
    %Get a fresh one for each trial
    R_AHVbyHDSpikes = zeros(num_AHV_bins, num_HD_bins); %array for the spikecounts
    R_AHVbyHDDwell = zeros(num_AHV_bins, num_HD_bins); %array for dwelltime
    
    for AHVindex = 1:num_AHV_bins
        
        for HDindex = 1:num_HD_bins
            
            %find time bins where both variables are within bin range
            conjunctive_timebins = find(AngHeadVel>=R_AHV_binedges(AHVindex) & AngHeadVel<R_AHV_binedges(AHVindex+1)...
                & HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1) & AngHeadVel~=0);
            
            R_AHVbyHDDwell(AHVindex, HDindex) = R_AHVbyHDDwell(AHVindex, HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
            
            for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                
                R_AHVbyHDSpikes(AHVindex, HDindex) = R_AHVbyHDSpikes(AHVindex, HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
            end
            
        end
    end
    
    %Plottng R AHV stuff
        
        figure('units','normalized','outerposition',[0 0 1 1]);
        rh = subplot(3,6,[4,5,10,11]);
        smoothing_kernel = 5;
        new_R_AHVbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
        new_R_AHVbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyHDSpikes);
        
        %         new_R_AHVbyHDDwell = R_AHVbyHDDwell; %no longer smoothing
        %         new_R_AHVbyHDSpikes = R_AHVbyHDSpikes;
        
        
        R_AHVbyHDRates = new_R_AHVbyHDSpikes./new_R_AHVbyHDDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_R_AHVbyHDDwell(:))/ numel(new_R_AHVbyHDDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_R_AHVbyHDDwell<bin_exclusion_criterion;
        R_AHVbyHDRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(R_AHV_vals,3:6:357);
        surf(X,Y,R_AHVbyHDRates');
        shading interp;
        grid off;
        max_right_heat_plot = max(R_AHVbyHDRates(:));
        cbar = [0,max_right_heat_plot];
        caxis(cbar);
        colormap(jet);
        ylim([0,360]);
        xlim([R_AHV_binedges(1),ceil(R_AHV_binedges(end-1))]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        new_session_text = strrep(session_text{session},'.','');
        new_session_text = strrep(new_session_text,'\','_');
        title(new_session_text);
        set(gca,'Fontsize',12);
        set(gca,'Yaxislocation','right');
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        
        R_AHV_Spikes = sum(R_AHVbyHDSpikes,2);
        R_AHV_Dwell = sum(R_AHVbyHDDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        R_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
        
        %No longer filtering Dwell!
        R_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Spikes);
        
        R_AHV_Rates = R_AHV_Spikes./R_AHV_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(R_AHV_Dwell(:))/ numel(R_AHV_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = R_AHV_Dwell<bin_exclusion_criterion & R_AHV_Spikes>2;
        R_AHV_Rates(bin_exclusion_filter) = NaN;
        
        rahv = subplot(3,6,[16,17]);
        %Adding shuffle for running speed
        shuffled_R_AHV_rates = zeros(1000,numel(R_AHV_Rates));
        
        for shuffle = 1:1000
            shuffled_R_AHV_byHDSpikes = shuffle_conjunctive(R_AHVbyHDSpikes);
            
            R_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_R_AHV_byHDSpikes,2));
            %R_AHV_Spikes =  sum(shuffled_R_AHV_byHDSpikes,2);
            R_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(R_AHVbyHDDwell,2));
            %No longer filtering Dwell!
            
            
            R_AHV_Rates_shuff = R_AHV_Spikes./R_AHV_Dwell;
            R_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
            shuffled_R_AHV_rates(shuffle,:) = R_AHV_Rates_shuff;
        end
        
        hold on
        boundedline(R_AHV_vals,nanmean(shuffled_R_AHV_rates,1),nanstd(shuffled_R_AHV_rates,1).*2);
        plot(R_AHV_vals,R_AHV_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        
        xlabel('R AHV (deg/s)');
        r_max_ahv_fr = max(R_AHV_Rates);
        ylim([0 Inf]);
        xlim([0,ceil(R_AHV_binedges(end-1))]);
        set(gca,'Fontsize',12);
        set(gca,'YColor','none');
        set(gca,'YTicklabel',[]);
        
        %Plotting just HD
        
        HD_Spk = sum(R_AHVbyHDSpikes,1);
        HD_Dwl = sum(R_AHVbyHDDwell,1);
        
        HD_Dwl = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Dwl); %this is the smoothing from GetRateMap function in UCL store
        HD_Spk = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Spk);
        
        HD_Rts = HD_Spk./HD_Dwl;
        
        bin_exclusion_criterion = 0.01 * (sum(HD_Dwl(:))/ numel(HD_Dwl(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = HD_Dwl<bin_exclusion_criterion;
        HD_Rts(bin_exclusion_filter) = NaN;
        
        rhd = subplot(3,6,[6,12]);
        plot(-177:6:177,HD_Rts, 'b', 'Linewidth', 2.0);
        ylabel('Firing Rate (Hz)');
        xlabel('HD - PFD (deg)');
        ylim([0 Inf]);
        r_max_hd_fr = max(Run_Rates);
        xlim([-180 180]);
        set(gca,'XTick',-180:90:180);
        set(gca,'Xaxislocation','top');
        view([90 90]); %orient correctly for subplotting
        set(gca,'XDir','reverse');
        set(gca,'YDir','reverse');
        box off;
        set(gca,'Fontsize',12);
       
        
        %% Now Dealing with L AHV
        
        %Get a fresh one for each trial
        L_AHVbyHDSpikes = zeros(num_AHV_bins, num_HD_bins); %array for the spikecounts
        L_AHVbyHDDwell = zeros(num_AHV_bins, num_HD_bins); %array for dwelltime
        
        for AHVindex = 1:num_AHV_bins
            
            for HDindex = 1:num_HD_bins
                
                %find time bins where both variables are within bin range
                conjunctive_timebins = find(AngHeadVel>=L_AHV_binedges(AHVindex) & AngHeadVel<L_AHV_binedges(AHVindex+1)...
                    & HeadDirection>=HD_binedges(HDindex) & HeadDirection<HD_binedges(HDindex+1) & AngHeadVel~=0);
                
                L_AHVbyHDDwell(AHVindex, HDindex) = L_AHVbyHDDwell(AHVindex, HDindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
                
                for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                    lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                    upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                    
                    L_AHVbyHDSpikes(AHVindex, HDindex) = L_AHVbyHDSpikes(AHVindex, HDindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
                end
                
            end
        end
        
        %Plottng L AHV stuff
        
        subplot(3,6,[2,3,8,9]);
        smoothing_kernel = 5;
        new_L_AHVbyHDDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyHDDwell); %this is the smoothing from GetRateMap function in UCL store
        new_L_AHVbyHDSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyHDSpikes);
        
        %         new_L_AHVbyHDDwell = L_AHVbyHDDwell; %no longer smoothing
        %         new_L_AHVbyHDSpikes = L_AHVbyHDSpikes;
        
        L_AHVbyHDRates = new_L_AHVbyHDSpikes./new_L_AHVbyHDDwell;
        
        bin_exclusion_criterion = 0.01 * (sum(new_L_AHVbyHDDwell(:))/ numel(new_L_AHVbyHDDwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = new_L_AHVbyHDDwell<bin_exclusion_criterion;
        L_AHVbyHDRates(bin_exclusion_filter) = NaN;
        
        [X,Y] = meshgrid(L_AHV_vals,3:6:357);
        surf(X,Y,L_AHVbyHDRates');
        shading interp;
        grid off;
         if max(L_AHVbyHDRates(:)) > max_right_heat_plot
            caxis(rh,[0,max(L_AHVbyHDRates(:))]); % set the clim on right heat map
            caxis([0,max(L_AHVbyHDRates(:))]); % set the clim on this heat map
            title(['Max FR = ',num2str(max(L_AHVbyHDRates(:))),' Hz']);
        else
            caxis(cbar); %use cbar from right heat map
            title(['Max FR = ',num2str(max_right_heat_plot),' Hz']);
        end
        colormap(jet);
        ylim([0,360]);
        xlim([ceil(L_AHV_binedges(2)),L_AHV_binedges(end)]);
        set(gca,'YTick',[]);
        set(gca,'YTickLabel',[]);
        set(gca,'XTick',[]);
        set(gca,'XTickLabel',[]);
        view([0 90]);
        set(gca,'Fontsize',12);
        set(gca,'XColor','none');
        set(gca,'YColor','none');
        
        L_AHV_Spikes = sum(L_AHVbyHDSpikes,2);
        L_AHV_Dwell = sum(L_AHVbyHDDwell,2);
        
        %NOTE: this filter kernel really effects the edges of the Dwell
        %distribution - is this a problem?
        L_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
        %No longer filtering Dwell!
        
        L_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Spikes);
        
        L_AHV_Rates = L_AHV_Spikes./L_AHV_Dwell;
        
        bin_exclusion_criterion = 0.01 * (sum(L_AHV_Dwell(:))/ numel(L_AHV_Dwell(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = L_AHV_Dwell<bin_exclusion_criterion & L_AHV_Spikes>2;
        L_AHV_Rates(bin_exclusion_filter) = NaN;
        
        subplot(3,6,[14,15]);
        %Adding shuffle for AHV
        shuffled_L_AHV_rates = zeros(1000,numel(L_AHV_Rates));
        
        for shuffle = 1:1000
            shuffled_L_AHV_byHDSpikes = shuffle_conjunctive(L_AHVbyHDSpikes);
            
            L_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_L_AHV_byHDSpikes,2));
            %L_AHV_Spikes =  sum(shuffled_L_AHV_byHDSpikes,2);
            L_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(L_AHVbyHDDwell,2));
            %No longer filtering Dwell!
            
            L_AHV_Rates_shuff = L_AHV_Spikes./L_AHV_Dwell;
            L_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
            shuffled_L_AHV_rates(shuffle,:) = L_AHV_Rates_shuff;
        end
        
        hold on
        boundedline(L_AHV_vals,nanmean(shuffled_L_AHV_rates,1),nanstd(shuffled_L_AHV_rates,1).*2);
        plot(L_AHV_vals,L_AHV_Rates,'b', 'Linewidth', 2.0);
        hold off
        
        ylabel('Firing Rate (Hz)');
        xlabel('L AHV (deg/s)');
         
        if max(L_AHV_Rates) > r_max_ahv_fr
            ylim([0, max(L_AHV_Rates)]);
            set(rahv,'ylim',[0, max(L_AHV_Rates)]);
        else
            ylim([0 r_max_ahv_fr]);
        end
        
        xlim([ceil(L_AHV_binedges(2)),0]);
        set(gca,'Fontsize',12);
        
        %Plotting just HD
        
        HD_Spk = sum(L_AHVbyHDSpikes,1);
        HD_Dwl = sum(L_AHVbyHDDwell,1);
        
        HD_Dwl = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Dwl); %this is the smoothing from GetRateMap function in UCL store
        HD_Spk = filter2(ones(smoothing_kernel)./smoothing_kernel,HD_Spk);
        
        HD_Rts = HD_Spk./HD_Dwl;
        
        bin_exclusion_criterion = 0.01 * (sum(HD_Dwl(:))/ numel(HD_Dwl(:)));  %1% of the total dwell across all trials
        bin_exclusion_filter = HD_Dwl<bin_exclusion_criterion;
        HD_Rts(bin_exclusion_filter) = NaN;
        
        subplot(3,6,[1,7]);
        plot(3:6:357,HD_Rts, 'b', 'Linewidth', 2.0);
        ylabel('Firing Rate (Hz)');
        xlim([0 360]);
        set(gca,'Xticklabel',[]);
        
         if max(HD_Rts) > r_max_hd_fr
            ylim([0, max(HD_Rts)]);
            set(rhd,'ylim',[0, max(HD_Rts)]);
        else
            ylim([0, r_max_hd_fr]);
         end
        
        set(gca,'Fontsize',12);
         view([90 90]); %orient correctly for subplotting
        set(gca,'XDir','reverse');
        set(gca,'YDir','reverse');
        set(gca,'XColor','none');
        box off;
        
    new_session_text = strrep(session_text{session},'.','');
    new_session_text = strrep(new_session_text,'\','_');
    saveas(gcf,[ new_session_text,'_conj_AHV_'],'tif');
    close(gcf);
    
    
    
    %% Now doing some kind of analysis of shift/ATI during head turns
    cd(startpath);
    
    ATIs(session) = time_slide_analysis(HeadDirection,SpikeTimes,prev_tmax,AngHeadVel);
    
    %% NOW DOING CONJUNCTIVE FOR THE TWO SPEEDS ON PER SESSION BASIS
    
    %Equalising dwell times here needs to be different for these conjunctive plots, because second
    %variable, given limited range, affects binning of first variable (unlike with HD).
    
    
    %Redoing binning of R_AHV and Running speed here based on
    %conjunctive limits......
    
    R_AHV = AngHeadVel(AngHeadVel>0 & speed>2);
    y =  quantile(R_AHV,num_AHV_bins-1);
    [counts, Id] = histc(R_AHV,[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
    R_AHV_binedges = [0;y(:);inf];
    R_AHV_vals = R_AHV_binedges(1:end-1);
    for idx = 1:numel(R_AHV_vals)-1 %ignore last, as it is infinity
        R_AHV_vals(idx) = mean(R_AHV_vals(idx:idx+1));
    end
    
    num_speed_bins = 20;
    y =  quantile(speed(speed>2 & AngHeadVel>0),num_speed_bins-1);
    [counts, Id] = histc(speed(speed>2),[-inf; y(:); inf]); %tester line to see if bin counts in speed dimension are equal (i.e. equal dwell)
    speed_binedges = [2;y(:);inf];
    speed_vals = speed_binedges(1:end-1);
    for idx = 1:numel(speed_vals)-1 %last speedval is just the binedges as upper edge is infinity
        speed_vals(idx) = (mean(speed_vals(idx:idx+1)));
    end
    
    
    %Get a fresh one for each trial
    R_AHVbyRunningSpeedSpikes = zeros(num_AHV_bins, num_speed_bins); %array for the spikecounts
    R_AHVbyRunningSpeedDwell = zeros(num_AHV_bins, num_speed_bins); %array for dwelltime
    
    for AHVindex = 1:num_AHV_bins
        
        for Speedindex = 1:num_speed_bins
            
            %find time bins where both variables are within bin range
            conjunctive_timebins = find(AngHeadVel>=R_AHV_binedges(AHVindex) & AngHeadVel<R_AHV_binedges(AHVindex+1)...
                & speed>=speed_binedges(Speedindex) & speed<speed_binedges(Speedindex+1) & AngHeadVel~=0);
            
            R_AHVbyRunningSpeedDwell(AHVindex, Speedindex) = R_AHVbyRunningSpeedDwell(AHVindex, Speedindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
            
            for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                
                R_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) = R_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
            end
            
        end
    end
    
    %Plottng R AHV stuff
    
    
    figure('units','normalized','outerposition',[0 0 1 1]);
    rh = subplot(3,6,[4,5,10,11]);
    smoothing_kernel = 5;
    new_R_AHVbyRunningSpeedDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyRunningSpeedDwell); %this is the smoothing from GetRateMap function in UCL store
    new_R_AHVbyRunningSpeedSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHVbyRunningSpeedSpikes);
    
    %         new_R_AHVbyRunningSpeedDwell = R_AHVbyRunningSpeedDwell; %no longer smoothing
    %         new_R_AHVbyRunningSpeedSpikes = R_AHVbyRunningSpeedSpikes;
    
    
    R_AHVbyRunningSpeedRates = new_R_AHVbyRunningSpeedSpikes./new_R_AHVbyRunningSpeedDwell;
    
    bin_exclusion_criterion = 0.01 * (sum(new_R_AHVbyRunningSpeedDwell(:))/ numel(new_R_AHVbyRunningSpeedDwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = new_R_AHVbyRunningSpeedDwell<bin_exclusion_criterion;
    R_AHVbyRunningSpeedRates(bin_exclusion_filter) = NaN;
    
    [X,Y] = meshgrid(R_AHV_vals,speed_vals);
    surf(X,Y,R_AHVbyRunningSpeedRates');
    shading interp;
    grid off;
    max_right_heat_plot = max(R_AHVbyRunningSpeedRates(:));
    top_speed_right = max(speed_vals);
    cbar = [0,max_right_heat_plot];
    caxis(cbar);
    colormap(jet);
    ylim([speed_binedges(1),ceil(speed_binedges(end-1))]);
    xlim([R_AHV_binedges(1),ceil(R_AHV_binedges(end-1))]);
    set(gca,'YTick',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'XTick',[]);
    set(gca,'XTickLabel',[]);
    view([0 90]);
    new_session_text = strrep(session_text{session},'.','');
    new_session_text = strrep(new_session_text,'\','_');
    title(new_session_text);
    set(gca,'Fontsize',12);
    set(gca,'Yaxislocation','right');
    set(gca,'XColor','none');
    set(gca,'YColor','none');
    
    R_AHV_Spikes = sum(R_AHVbyRunningSpeedSpikes,2);
    R_AHV_Dwell = sum(R_AHVbyRunningSpeedDwell,2);
    
    %NOTE: this filter kernel really effects the edges of the Dwell
    %distribution - is this a problem?
    R_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
    
    %No longer filtering Dwell!
    R_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,R_AHV_Spikes);
    
    R_AHV_Rates = R_AHV_Spikes./R_AHV_Dwell;
    
    bin_exclusion_criterion = 0.01 * (sum(R_AHV_Dwell(:))/ numel(R_AHV_Dwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = R_AHV_Dwell<bin_exclusion_criterion & R_AHV_Spikes>2;
    R_AHV_Rates(bin_exclusion_filter) = NaN;
    
    rahv = subplot(3,6,[16,17]);
    %Adding shuffle for running speed
    shuffled_R_AHV_rates = zeros(1000,numel(R_AHV_Rates));
    
    for shuffle = 1:1000
        shuffled_R_AHV_byRunningSpeedSpikes = shuffle_conjunctive(R_AHVbyRunningSpeedSpikes);
        
        R_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_R_AHV_byRunningSpeedSpikes,2));
        %R_AHV_Spikes =  sum(shuffled_R_AHV_byHDSpikes,2);
        R_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(R_AHVbyRunningSpeedDwell,2));
        %No longer filtering Dwell!
        
        
        R_AHV_Rates_shuff = R_AHV_Spikes./R_AHV_Dwell;
        R_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
        shuffled_R_AHV_rates(shuffle,:) = R_AHV_Rates_shuff;
    end
    
    hold on
    boundedline(R_AHV_vals,nanmean(shuffled_R_AHV_rates,1),nanstd(shuffled_R_AHV_rates,1).*2);
    plot(R_AHV_vals,R_AHV_Rates,'b', 'Linewidth', 2.0);
    hold off
    
    
    xlabel('R AHV (deg/s)');
    ylim([0 Inf]);
    r_max_ahv_fr = max(R_AHV_Rates);
    xlim([0,ceil(R_AHV_binedges(end-1))]);
    set(gca,'Fontsize',12);
    set(gca,'YColor','none');
    set(gca,'YTicklabel',[]);
    
    %Plotting just Running Speed
    
    Run_Spikes = sum(R_AHVbyRunningSpeedSpikes,1);
    Run_Dwell = sum(R_AHVbyRunningSpeedDwell,1);
    
    %NOTE: this filter kernel really effects the edges of the Dwell
    %distribution - is this a problem?
    Run_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Dwell); %this is the smoothing from GetRateMap function in UCL store
    %No longer filtering Dwell!
    
    Run_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Spikes);
    
    Run_Rates = Run_Spikes./Run_Dwell;
    
    bin_exclusion_criterion = 0.01 * (sum(Run_Dwell(:))/ numel(Run_Dwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = Run_Dwell<bin_exclusion_criterion;
    Run_Rates(bin_exclusion_filter) = NaN;
    
    rs = subplot(3,6,[6,12]);
    %Adding shuffle for running speed
    shuffled_run_rates = zeros(1000,numel(Run_Rates));
    
    for shuffle = 1:1000
        shuffled_RunbyHDSpikes = shuffle_conjunctive(RunningSpeedbyHDSpikes);
        
        Running_Speed_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_RunbyHDSpikes,2));
        %Running_Speed_Spikes = sum(shuffled_RunbyHDSpikes,2);
        
        Running_Speed_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(RunningSpeedbyHDDwell,2));
        
        %No longer filtering Dwell!
        %Running_Speed_Dwell = Run_Dwell;
        
        RunSpeed_Rates = Running_Speed_Spikes./Running_Speed_Dwell;
        RunSpeed_Rates(bin_exclusion_filter) = NaN;
        shuffled_run_rates(shuffle,:) = RunSpeed_Rates;
    end
    
    hold on
    boundedline(speed_vals,nanmean(shuffled_run_rates,1),nanstd(shuffled_run_rates,1).*2);
    plot(speed_vals,Run_Rates,'b', 'Linewidth', 2.0);
    hold off
    
    ylabel('Firing Rate (Hz)');
    xlabel('Speed (cm/s)');
    ylim([0 Inf]);
    r_max_speed_fr = max(Run_Rates);
    xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
    set(gca,'Xaxislocation','top');
    view([90 90]); %orient correctly for subplotting
    set(gca,'XDir','reverse');
    set(gca,'YDir','reverse');
    set(gca,'Fontsize',12);
    
    %% Now doing running speed by L AHV
    %Equalising dwell times here needs to be different for these conjunctive plots, because second
    %variable, given limited range, affects binning of first variable (unlike with HD).
    
    %Redoing binning of R_AHV and Running speed here based on
    %conjunctive limits......
    
    L_AHV = AngHeadVel(AngHeadVel<0 & speed>2);
    y =  quantile(abs(L_AHV),num_AHV_bins-1);
    [counts, Id] = histc(abs(L_AHV),[0; y(:); inf]); %tester line to see if bin counts in AHV dimension are equal (i.e. equal dwell)
    L_AHV_binedges = [0;y(:);inf];
    L_AHV_vals = L_AHV_binedges(1:end-1);
    for idx = 1:numel(L_AHV_vals)-1 %ignore last, as it is infinity
        L_AHV_vals(idx) = mean(L_AHV_vals(idx:idx+1));
    end
    %L AHV is all negative so make the vals etc negative
    L_AHV_binedges = flipud(-L_AHV_binedges);
    L_AHV_vals = flipud(-L_AHV_vals);
    
    num_speed_bins = 20;
    y =  quantile(speed(speed>2 & AngHeadVel<0),num_speed_bins-1);
    [counts, Id] = histc(speed(speed>2),[-inf; y(:); inf]); %tester line to see if bin counts in speed dimension are equal (i.e. equal dwell)
    speed_binedges = [2;y(:);inf];
    speed_vals = speed_binedges(1:end-1);
    for idx = 1:numel(speed_vals)-1 %last speedval is just the binedges as upper edge is infinity
        speed_vals(idx) = (mean(speed_vals(idx:idx+1)));
    end
    
    
    %Get a fresh one for each trial
    L_AHVbyRunningSpeedSpikes = zeros(num_AHV_bins, num_speed_bins); %array for the spikecounts
    L_AHVbyRunningSpeedDwell = zeros(num_AHV_bins, num_speed_bins); %array for dwelltime
    
    for AHVindex = 1:num_AHV_bins
        
        for Speedindex = 1:num_speed_bins
            
            %find time bins where both variables are within bin range
            conjunctive_timebins = find(AngHeadVel>=L_AHV_binedges(AHVindex) & AngHeadVel<L_AHV_binedges(AHVindex+1)...
                & speed>=speed_binedges(Speedindex) & speed<speed_binedges(Speedindex+1) & AngHeadVel~=0);
            
            L_AHVbyRunningSpeedDwell(AHVindex, Speedindex) = L_AHVbyRunningSpeedDwell(AHVindex, Speedindex) + (numel(conjunctive_timebins)/ 50);   %number of 20ms time bins, so /50 to get time in seconds
            
            for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
                lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
                upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
                
                L_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) = L_AHVbyRunningSpeedSpikes(AHVindex, Speedindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
            end
            
        end
    end
    
    %Plottng L AHV stuff
    
    
    subplot(3,6,[2,3,8,9]);
    smoothing_kernel = 5;
    new_L_AHVbyRunningSpeedDwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyRunningSpeedDwell); %this is the smoothing from GetRateMap function in UCL store
    new_L_AHVbyRunningSpeedSpikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHVbyRunningSpeedSpikes);
    
    %         new_L_AHVbyRunningSpeedDwell = L_AHVbyRunningSpeedDwell; %no longer smoothing
    %         new_L_AHVbyRunningSpeedSpikes = L_AHVbyRunningSpeedSpikes;
    
    
    L_AHVbyRunningSpeedRates = new_L_AHVbyRunningSpeedSpikes./new_L_AHVbyRunningSpeedDwell;
    
    bin_exclusion_criterion = 0.01 * (sum(new_L_AHVbyRunningSpeedDwell(:))/ numel(new_L_AHVbyRunningSpeedDwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = new_L_AHVbyRunningSpeedDwell<bin_exclusion_criterion;
    L_AHVbyRunningSpeedRates(bin_exclusion_filter) = NaN;
    
    [X,Y] = meshgrid(L_AHV_vals,speed_vals);
    surf(X,Y,L_AHVbyRunningSpeedRates');
    shading interp;
    grid off;
    %cbar=([0 max(L_AHVbyRunningSpeedRates(:))]);
    if max(L_AHVbyRunningSpeedRates(:)) > max_right_heat_plot
        caxis(rh,[0,max(L_AHVbyRunningSpeedRates(:))]); % set the clim on right heat map
        caxis([0,max(L_AHVbyRunningSpeedRates(:))]); % set the clim on this heat map
        title(['Max FR = ',num2str(max(L_AHVbyRunningSpeedRates(:))),' Hz']);
    else
        caxis(cbar); %use cbar from right heat map
        title(['Max FR = ',num2str(max_right_heat_plot),' Hz']);
    end
    
    colormap(jet);
    %Also need to do pratting around with heat maps y limit for running
    %speed....
    if speed_binedges(end-1) < top_speed_right
        ylim([speed_binedges(1),ceil(speed_binedges(end-1))]);
        set(rh,'ylim',[speed_binedges(1),ceil(speed_binedges(end-1))]);
    else
        ylim([speed_binedges(1),top_speed_right]);
    end
    xlim([ceil(L_AHV_binedges(2)),0]);
    set(gca,'YTick',[]);
    set(gca,'YTickLabel',[]);
    set(gca,'XTick',[]);
    set(gca,'XTickLabel',[]);
    view([0 90]);
    
    set(gca,'Fontsize',12);
    set(gca,'XColor','none');
    set(gca,'YColor','none');
    
    L_AHV_Spikes = sum(L_AHVbyRunningSpeedSpikes,2);
    L_AHV_Dwell = sum(L_AHVbyRunningSpeedDwell,2);
    
    %NOTE: this filter kernel really effects the edges of the Dwell
    %distribution - is this a problem?
    L_AHV_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Dwell); %this is the smoothing from GetRateMap function in UCL store
    
    %No longer filtering Dwell!
    L_AHV_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,L_AHV_Spikes);
    
    L_AHV_Rates = L_AHV_Spikes./L_AHV_Dwell;
    
    bin_exclusion_criterion = 0.01 * (sum(L_AHV_Dwell(:))/ numel(L_AHV_Dwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = L_AHV_Dwell<bin_exclusion_criterion & L_AHV_Spikes>2;
    L_AHV_Rates(bin_exclusion_filter) = NaN;
    
    subplot(3,6,[14,15]);
    %Adding shuffle for running speed
    shuffled_L_AHV_rates = zeros(1000,numel(L_AHV_Rates));
    
    for shuffle = 1:1000
        shuffled_L_AHV_byRunningSpeedSpikes = shuffle_conjunctive(L_AHVbyRunningSpeedSpikes);
        
        L_AHV_Spikes =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_L_AHV_byRunningSpeedSpikes,2));
        %L_AHV_Spikes =  sum(shuffled_L_AHV_byHDSpikes,2);
        L_AHV_Dwell =  filter2(ones(smoothing_kernel)./smoothing_kernel,sum(L_AHVbyRunningSpeedDwell,2));
        %No longer filtering Dwell!
        
        
        L_AHV_Rates_shuff = L_AHV_Spikes./L_AHV_Dwell;
        L_AHV_Rates_shuff(bin_exclusion_filter) = NaN;
        shuffled_L_AHV_rates(shuffle,:) = L_AHV_Rates_shuff;
    end
    
    hold on
    boundedline(L_AHV_vals,nanmean(shuffled_L_AHV_rates,1),nanstd(shuffled_L_AHV_rates,1).*2);
    plot(L_AHV_vals,L_AHV_Rates,'b', 'Linewidth', 2.0);
    hold off
    
    ylabel('Firing Rate (Hz)');
    xlabel('L AHV (deg/s)');
    
    if max(L_AHV_Rates) > r_max_ahv_fr
        ylim([0 max(L_AHV_Rates)]);
        set(rahv,'ylim',[0 max(L_AHV_Rates)]);
    else
        ylim([0 r_max_ahv_fr]);
    end
    xlim([ceil(L_AHV_binedges(2)),0]);
    set(gca,'Fontsize',12);
    
    %Plotting just Running Speed
    
    Run_Spikes = sum(L_AHVbyRunningSpeedSpikes,1);
    Run_Dwell = sum(L_AHVbyRunningSpeedDwell,1);
    
    %NOTE: this filter kernel really effects the edges of the Dwell
    %distribution - is this a problem?
    Run_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Dwell); %this is the smoothing from GetRateMap function in UCL store
    %No longer filtering Dwell!
    
    Run_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,Run_Spikes);
    
    Run_Rates = Run_Spikes./Run_Dwell;
    
    bin_exclusion_criterion = 0.01 * (sum(Run_Dwell(:))/ numel(Run_Dwell(:)));  %1% of the total dwell across all trials
    bin_exclusion_filter = Run_Dwell<bin_exclusion_criterion;
    Run_Rates(bin_exclusion_filter) = NaN;
    
    subplot(3,6,[1,7]);
    %Adding shuffle for running speed
    shuffled_run_rates = zeros(1000,numel(Run_Rates));
    
    for shuffle = 1:1000
        shuffled_RunbyHDSpikes = shuffle_conjunctive(RunningSpeedbyHDSpikes);
        
        Running_Speed_Spikes = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(shuffled_RunbyHDSpikes,2));
        %Running_Speed_Spikes = sum(shuffled_RunbyHDSpikes,2);
        
        Running_Speed_Dwell = filter2(ones(smoothing_kernel)./smoothing_kernel,sum(RunningSpeedbyHDDwell,2));
        
        %No longer filtering Dwell!
        %Running_Speed_Dwell = Run_Dwell;
        
        RunSpeed_Rates = Running_Speed_Spikes./Running_Speed_Dwell;
        RunSpeed_Rates(bin_exclusion_filter) = NaN;
        shuffled_run_rates(shuffle,:) = RunSpeed_Rates;
    end
    
    hold on
    boundedline(speed_vals,nanmean(shuffled_run_rates,1),nanstd(shuffled_run_rates,1).*2);
    plot(speed_vals,Run_Rates,'b', 'Linewidth', 2.0);
    hold off
    
    ylabel('Firing Rate (Hz)');
    %xlabel('Speed (cm/s)');
    set(gca,'Xticklabel',[]);
    ylim([0 Inf]);
    
    if max(Run_Rates) > r_max_speed_fr
        ylim([0, max(Run_Rates)]);
        set(rs,'ylim',[0, max(Run_Rates)]);
    else
        ylim([0, r_max_speed_fr]);
    end
    
    if speed_binedges(end-1) < top_speed_right
        xlim([speed_binedges(1),ceil(speed_binedges(end-1))]);
        set(rs,'ylim',[speed_binedges(1),ceil(speed_binedges(end-1))]);
    else
        xlim([speed_binedges(1),top_speed_right]);
    end
    
    set(gca,'Fontsize',12);
    view([90 90]); %orient correctly for subplotting
    set(gca,'XDir','reverse');
    set(gca,'YDir','reverse');
    set(gca,'XColor','none');
    
    
    cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/conjunctive_plots/per_session_plots'); %save in the plots directory
    
    new_session_text = strrep(session_text{session},'.','');
    new_session_text = strrep(new_session_text,'\','_');
    saveas(gcf,[new_session_text,'_conj_speed_AHV'],'tif');
    close(gcf);
    
    fclose('all');
    
    cd(startpath); %GO HOME
    
    
    
    
end


%% SAVE AREA STATS
cd('/Volumes/ibn/Jeffery lab/Hector Page/Yave data/HectorsNewAnalysis/conjunctive_plots');


%% saving all the stats for collapsed sessions
if strcmpi(mode,'first')
    %save data needed
    save('ATIs1.mat','ATIs');
elseif strcmpi(mode,'second')
    %save data needed
    save('ATIs2.mat','ATIs');
else
    %save data needed
    save('ATIs3.mat','ATIs');
end
cd(startpath);
end

function shuffled_array = shuffle_conjunctive(input_array)
%This function keeps the column sum the same for the HD tuning, but
%shuffles the spikes in that column with respect to speed....

%This shuffles spikecount with respect to speed, but keeps HD tuning
%unaffected


shuffled_array = zeros(size(input_array));

%Rejected over-engineered version below
%column_sums = sum(input_array,1);

% for idx = 1:length(column_sums);
%     s = column_sums(idx); %target sum for this
%     m = size(shuffled_array,1); %this is the height of a column
%
%     a = ones(1,m);
%     p = m;
%     for k = s-1:-1:1
%         if rand < (p-1)/k
%             p = p - 1;
%         else
%             a(p) = a(p) + 1;
%         end
%     end
%
%     shuffled_array(:,idx) = a;
% end

for idx = 1:size(input_array,2)
    shuffled_array(:,idx) = input_array(randperm(size(input_array,1)),idx);
end


end







