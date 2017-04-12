function new_analysis(mode,flip_cells, Trial)
close all;

%% SETTING UP DIRECTORIES

startpath = pwd; %come back here afterwards
%Add all the necessary functions to path
addpath(genpath('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/RoddyCodes'));

%%Setting up analysis targets

if(~strcmpi(mode,'file'))    %manual analysis target entry
    
    analysis_targets = zeros(1,4); %this holds targets of analysis in format: Rat Date Tetrode Cell
    entry_finished = 'N';
    target_row_index =  0;
    disp('Selecting Data to analyse, multiple trials/tetrodes/cell from same rat requires repeat entry');
    while(strcmp(entry_finished,'N'))
        target_row_index = target_row_index+1;
        prompt = 'Select Rat # (omit letter R):';
        analysis_targets(target_row_index,1) = input(prompt);
        
        prompt = 'Select Date in format ddmmyy :';
        analysis_targets(target_row_index,2) = input(prompt);
        
        prompt = 'Select Tetrode:';
        analysis_targets(target_row_index,3) = input(prompt);
        
        prompt = 'Select Cell:';
        analysis_targets(target_row_index,4) = input(prompt);
        
        prompt = 'Are you done? (Y/N)';
        entry_finished = input(prompt, 's');
        
    end
    
    [total_plots, ~] = size(analysis_targets);
else                                    %read in analysis targets from file
    prompt = 'Input filename in single quotes with extension:';      %need to be in correct directory
    
    input_file = strcat('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/',input(prompt));
    
    %now read in targets for anaylsis
    [~,text,~] = xlsread(input_file);
    
    total_plots = numel(text);
    
    analysis_targets = zeros(total_plots,4); %this holds targets of analysis in format: Rat Date Tetrode Cell
    
    for target = 1:total_plots
        [analysis_targets(target,1:4),~] = sscanf(text{target},'R%f_%f_tet%f_cell%f');
    end
    
    
    prompt = 'Give Input file listing cell types:';
    
    celltype_input = strcat('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/',input(prompt));
    
    %now read in cell tpyes
    cell_type = xlsread(celltype_input);
    
end

%savepath = '/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/PFD_ISI';

% running_speed_stats = zeros(total_plots,3); %stats in the format slope intercept R^2
% AHV_stats = zeros(total_plots,3); %as above for AHV
% absAHV_stats = zeros(total_plots,3); %as above for absolute AHV
%
% outsidePFDspikes = zeros(total_plots,1);
% peaks = zeros(total_plots,2);

%for LFP analysis
% LFP_frequency = zeros(total_plots,2); %frequency, and then power
% speed_theta_stats = zeros(total_plots,4); %in format, slope-intercept-r-p
%theta_ratio = zeros(total_plots,1);
%alt_theta_ratio = zeros(total_plots,1);
%intrinsic_freq_stats = zeros(total_plots,4);

%  ISI_peaks = zeros(total_plots,2); %in format, half peak time onwards from peak, peak time
%  ISI_exp_params = zeros(total_plots,2); %in format coeff1 coeff2

% %These are data structures to hold ISIs
% within_ISIs = 0;
% between_ISIs = 0;
% RSCHD_ISIs = 0;
% 
% within_peak = 0;
% within_half_peak = 0;
% within_constant_a = 0;
% within_constant_b = 0;
% 
% between_peak = 0;
% between_half_peak = 0;
% between_constant_a = 0;
% between_constant_b = 0;
% 
% HD_peak = 0;
% HD_half_peak = 0;
% HD_constant_a = 0;
% HD_constant_b = 0;
% 
% %Data struture to count number of ISI observations
% within_spikecount = 0;
% between_spikecount = 0;
% HD_spikecount = 0;
% 
% %Holding histogram bin means
% bin_counts_per_cell = zeros(total_plots,500);

for plotnumber = 1:total_plots %do all the plotting for each set of data
    
    %Trial = 1;  %first plot of movement correlates is trial 1
    Rat = analysis_targets(plotnumber,1);
    Date = analysis_targets(plotnumber,2);
    tetrode_1 = analysis_targets(plotnumber,3);
    cell_1 = analysis_targets(plotnumber,4);
    
    
    disp('------------------------------------');
    disp(['Progress = ', num2str(((plotnumber-1)/total_plots)*100),'% (',num2str(plotnumber),' of ',num2str(total_plots),' plots)']);
    disp('------------------------------------');
    
    
    
    filepath = ['/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/Context box recordings/R',num2str(Rat)];
    
    if(plotnumber>198) %just dealing with R636 data for RSC hd cells
        filepath = ['/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/Context box recordings/R',num2str(Rat),'/_previous'];
    end
    
    if(plotnumber>113 && plotnumber<117) %just dealing with R636 data for RSC flip cells
        filepath = ['/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/Context box recordings/R',num2str(Rat),'/_previous'];
    end
    
    %  if(plotnumber>198 && plotnumber< 209) %just dealing with R636 cells for this rsc hd cells
    %      filepath = ['/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/Context box recordings/R',num2str(Rat),'/_previous'];
    %  end
    
    % if(plotnumber>=62 && plotnumber<= 74) %FOR THETA RATIO ONLY
    %       filepath = ['/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/Context box recordings/R',num2str(Rat),'/_previous'];
    % end
    
    cd(filepath);
    
    if(Date<100000) %always add a zero in front of the data where a number <10 would be for the correct day format
        DateString = strcat(num2str(0),num2str(Date));
    else
        DateString = num2str(Date);
    end
    
    filename = ['R',num2str(Rat),'_',DateString,'_T',num2str(Trial)];
    full_filename = strcat(filename,'.set');
    %% SETTING UP DATA
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%% Create data folders for later and save inputs
    % disp('Making data folders...')
    % if ~exist('Figures', 'dir')	                                                                                                        		% if a folder with that name does not exist in the current directory
    %         disp(sprintf('\t...making Figures folder'));                                                                 					% display a message
    %         mkdir('Figures');                                                                                                            			% make a new folder with the current folder name
    % end % if ~exist('Figures', 'dir')
    % if ~exist('Data', 'dir')	                                                                                                        		% if a folder with that name does not exist in the current directory
    %         disp(sprintf('\t...making Data folder'));                                                                 					% display a message
    %         mkdir('Data');                                                                                                            			% make a new folder with the current folder name
    % end % if ~exist('Data', 'dir')
    % disp(sprintf('\t...done'));
    % disp('------------------------------------');
    %%%%%%%%%%%%%% Run or fetch AllDACQdata
    
    % disp('------------------------------------');
    % disp(sprintf('Fetching DACQ data...'));
    % if ~exist('Data/all_data.mat');
    %         disp(sprintf('\t...running readAllDACQdata'));
    % 	mtint = readAllDACQdata(filepath,full_filename);
    % 	save('Data/all_data.mat','mtint');
    % else
    %         disp(sprintf('\t...loading saved data'));
    % 	load('Data/all_data.mat','mtint');
    % end % if ~exist('all_data.mat')
    % disp(sprintf('\t...done'));
    % disp('------------------------------------');
    
    disp('------------------------------------');
    disp(sprintf('Fetching DACQ data...'));
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
    
    Spikes = mtint.tetrode(tetrode_1).cut==cell_1;                                         %logical index of which spikes on this tetrode are cell_1
    SpikeTimes= mtint.tetrode(tetrode_1).ts(Spikes);                                       %indexing in to extract spiketimes for this cell
    
    %position data
    clu_assign = mtint.tetrode(tetrode_1).cut;												% clu_assign is a vector of numbers, one for each spike, each number corresponds to a cluster
    pos_assign = mtint.tetrode(tetrode_1).pos_sample;                                       % pos_assign is a vector of numbers, one for each spike, each number corresponds to a position data point
    n_spikes = length(find(clu_assign==cell_1));
    
    frate = n_spikes ./ tmax;                                                               %average firing rate over trial
    
    pox = double(posx);
    poy = double(posy);
    
    spx = double(posx(pos_assign(clu_assign == cell_1)));
    spy = double(posy(pos_assign(clu_assign == cell_1)));
    
    hd = double(head_direct(pos_assign(clu_assign == cell_1)));
    
    com_min_x = min(pox);
    com_min_y = min(-poy);
    pox = pox - com_min_x;
    poy = -poy - com_min_y;
    spx = spx - com_min_x;
    spy = -spy - com_min_y;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% SETTING UP FIGURE
    % 				fig_one = figure('Visible','on');
    % 				set(gcf,'PaperUnits','centimeters'); 						% This sets the units of the current figure (gcf = get current figure) on paper to centimeters.
    % 				xSize = 30; 												% These are my size variables, width of 30 and a height of 21, will be used a lot later.
    % 				ySize = 28;
    % 				xLeft = (30-xSize)/2; 											% Additional coordinates to center the figure on A4-paper
    % 				yTop = (28-ySize)/2;
    % 				set(gcf,'PaperPosition',[xLeft yTop xSize ySize]);							% This command sets the position and size of the figure on the paper to the desired values.
    % 				set(gcf,'Position',[5 5 xSize*50 ySize*50]); 								% This additional command resizes and repositions the figure window on the computer screen, so that it has the same aspect ratio as the final figure.
    %
    % 		                fig_ver = 8;			                                                                        % Number of rows for subplots
    % 		                fig_hor = 6;			                                                                        % Number of columns for subplots
    %
    % 		                pos_tile = [1 2 7 8]; 	                                                                                % Mehalanobis distance plot tiles
    % 		                chrono_tile = [3 4 9 10]; 	                                                                        % Dwell time pcolor plot tiles
    % 		                place_tile = [5 6 11 12];		                                                                        % Pcolor place field plot
    %
    % 		                HD_tile = [13 14 19 20];
    % 		                HD_tile2 = [15 16 21 22];
    % 		                AutoCorr_tile = [17 18 23 24];
    %
    % 		                ch_tiles = [25;26;31;32];			                                                                % Waveforms tiles
    % 		                thetalin_tile = [27 28 33 34];		                                                                        % 200ms bar plot tiles
    % 		                spik_tile = [29 30 35 36]; 		                                                                        % Spike histogram tiles
    %
    % 		                meh_tile = [37 38 43 44];		                                                                        % position plot tiles
    % 		                theta_tile = [39 40 45 46];		                                                                        % 200ms bar plot tiles
    % 		                refrac_tile = [41 42 47 48];		                                                                        % 20ms bar plot tiles
    %%
    %%%%%%%%%%%%%%% Plot: Spikes vs Time
    %subplot(fig_ver,fig_hor,spik_tile); % top left lower: spike histogram
    % 			        tstart = 0;
    % 			        tstop = tmax;
    % 			        tstop = tstop + 5; 														% padding at the end for stray spikes
    % 			        timeh = hist(SpikeTimes,ceil(tstop - tstart));
    %bar(floor(tstart):(floor(tstart) + ceil(tstop -tstart) -1), timeh, 0.9, 'k');
    % 			        ylabel('Frate (Hz)', 'FontSize', 6);
    %                     set(gca,'YAxisLocation','right');
    %
    % 			        axis([tstart tstop 0 max(timeh)*1.1]);
    %
    % 			        set(gca, 'FontSize', 6);
    %                     set(gca,'LineWidth',2,'layer','top');
    % 			        xlabel('Time (s)', 'FontSize', 6);
    
    
    %%
    %%%%%%%%%%%%%%%%%% Plot: Position plot
    %                 figure();
    % 				plot(pox,poy,'k')
    % 				hold on
    % 				plot(spx,spy,'r.','MarkerSize',15)
    %
    %                                 axis([min(min(pox)) max(max(pox)) min(min(poy)) max(max(poy))]);
    %                                 set(gca, 'DataAspectRatio', [1 1 1]);
    %                                 set(gca, 'FontSize', 6);
    %                                 title(sprintf('Spikes: %.f',n_spikes),'FontSize', 8);
    %                                 axis off;
    %%
    %%%%%%%%%%%%%%%%%%% Plot: Head Direction (HD) Polar plot
    Vars=daVarsStruct;
    in_polar1 = make_in_struct_for_rm (mtint,tetrode_1,cell_1,50,Vars.rm.smthKernPos,Vars.rm.binSizePosCm,Vars.rm.binSizeDir,0,Vars,'dir');
    if size(in_polar1.posDir,2) ~= 1
        in_polar1.posDir = in_polar1.posDir';
    end
    in = in_polar1;
    PolarMap_cell1 = GetRateMap (in);
    circRm_cell1=PolarMap_cell1.map;
    
    %% KDE plotting
    
    %create a spike train in MS
    num_bins = 600*1000; %note, only works for Pierre's trial data, other situations use line above?
    train = zeros(num_bins,1); %creates an array up to no_bins, 1 element per ms, with 1=spike, 0=no spike

    for spk = 1:numel(SpikeTimes) %for each spike
      bin = round(SpikeTimes(spk)*1000) ; %bin where this spike occurs, rounded to nearest MS
      train(bin) = train(bin) + 1; %put a 1 in the spike train at this spike's location
    end

    HeadDirection = in.posDir; %HD at 50ms sample rate
    MS_HD = interp1(20:20:600000,HeadDirection,1:600000);
    
    MS_HD = MS_HD * (pi/180); %get in radians
    
    MS_HD = MS_HD(20:end);
    train = train(20:end);
    
    SpikeDirs = MS_HD(logical(train));
    
    samples = 0: pi/30 : 2*pi;
    SpikeVFEstimate = circ_ksdensity(SpikeDirs,samples);
    DwellVFEstimate = circ_ksdensity(MS_HD,samples);
    
  
    
        figure()
        subplot(3,1,1);
        p = mmpolar(samples,SpikeVFEstimate);
         set(p(1),'LineWidth',2,'color','b');
        title('Spike KDE');
       

        subplot(3,1,2);
        p = mmpolar(samples,DwellVFEstimate);
         set(p(1),'LineWidth',2,'color','b');
         title('Dwell KDE');
      

         subplot(3,1,3);
        p = mmpolar(samples,SpikeVFEstimate./DwellVFEstimate);
         set(p(1),'LineWidth',2,'color','b');
         title('Rate KDE');
         
         
         %Using linear KDE rather than circular as above, function from
         %file exchange
         
         [alt_SpikeVFEstimate, ~, ~] = ksdensity(SpikeDirs,samples);%[bandwidth,density,xmesh,cdf]=kde(data,n,MIN,MAX);
         [alt_DwellVFEstimate, ~, ~] = ksdensity(MS_HD,samples);%[bandwidth,density,xmesh,cdf]=kde(data,n,MIN,MAX);
         
         figure()
        subplot(3,1,1);
        plot(samples,alt_SpikeVFEstimate,'r','LineWidth',2.0);
        hold on
        plot(samples,SpikeVFEstimate,'b--','LineWidth',2.0);
        title('Spike KDE');
        xlim([0, 2*pi]);
        ylabel('Density');
        xlabel('Angle in Radians');
       

        subplot(3,1,2);
         plot(samples,alt_DwellVFEstimate,'r','LineWidth',2.0);
         hold on
         plot(samples,DwellVFEstimate,'b--','LineWidth',2.0);
         title('Dwell KDE');
         xlim([0, 2*pi]);
          ylabel('Density');
        xlabel('Angle in Radians');

         subplot(3,1,3);
         plot(samples,alt_SpikeVFEstimate./alt_DwellVFEstimate,'r','LineWidth',2.0);
         hold on
         plot(samples,SpikeVFEstimate./DwellVFEstimate,'b--','LineWidth',2.0);
         title('Rate KDE');
         xlim([0, 2*pi]);
          ylabel('Density');
        xlabel('Angle in Radians');
   
    
    
    
    samples = 0: pi/30 : 2*pi;
    vfestimate = circ_ksdensity(circRm_cell1,samples);
    mmpolar(samples,vfestimate,...
     'Grid','off','RTickVisible','off','RTickLabelVisible','off','TTickLabelVisible',...
     'off','TTickVisible','off', 'Color','TTickSign','+','Border','off','Axis','off');
    
    %%
    %subplot(fig_ver,fig_hor,HD_tile);
    figure()
     p = mmpolar(0: pi/30 : 2*pi,[circRm_cell1' circRm_cell1(1)'],...
     'Grid','off','RTickVisible','off','RTickLabelVisible','off','TTickLabelVisible',...
     'off','TTickVisible','off', 'Color','TTickSign','+','Border','off','Axis','off');
    
    hold on
    line([0 0],[-1 1],'color','k');
    line([-1 1],[0 0],'color','k');
    hold off
    
    set(p(1),'LineWidth',2,'color','b');
    
    %Plot: Regular Head Direction (HD) line plot
    %subplot(fig_ver,fig_hor,HD_tile2);
    plot(circRm_cell1,'b','linewidth',2);
    hold off
    xlim([1,60]);
    set(gca, 'XTick', [10:10:60]);
    set(gca,'xticklabel',[60:60:360]);
    ylim([0,ceil(max(circRm_cell1))]);
    
    
    %%
    % %%%%%%%%%%%%%%% Plot: Place field (firing rate map) plot
    % bin_size = 5; 												               % (cm), for calculating the rate map.
    % sigma = 2;                                                                 % sigma (gaussian standard deviation) to be used for rate and position map smoothing
    % min_dwell = 0.00001;                                                       % total number of seconds that rat has to be in a bin for it to count
    %
    %
    %
    % % subplot(fig_ver,fig_hor,place_tile); % bottom right: place field				       % Adjusting values
    %                                 %[posmap,frmap,xmap,ymap,hdmap,spdmap] = pfield_s3_pos(posx_,posy_,orig_hd, posxst, posyst, bin_size, pixel_ratio, min_dwell_distance, min_dwell_time, dt_position, smooth, 1);
    % 				[posmap spikmap frmap] = Mapper(poy,pox,spy,spx,pixel_ratio,bin_size,min_dwell,sigma); 					% Roddy's mapping script
    % 				frmap = fliplr(frmap);
    % 				posmap = fliplr(posmap);
    % 				spikmap = fliplr(spikmap);
    %
    %                                 skaggs = skaggs_info2(frmap,posmap);
    %                                 spars = sparsity(frmap, posmap);
    %                                 cohe = spatial_coherence(frmap, posmap);
    %
    %                                 max_firing = max(max(frmap));
    % 				min_fir = min(min(frmap));
    % 				back_fir = mean(nanmean(frmap));
    % 				max_fir = max(max(frmap));
    % 				[pfield_dataout] = CountPfields3(frmap,posmap);
    %
    % 				% setup colourbar
    %
    % 	                        	if max_fir < 1
    % 	                        		cbar = [0 1]; 							% if  maximum is less than 1Hz then set maximum to 1Hz
    % 	                        	else
    % 	                        		cbar = [0 max_fir];						% otherwise use the actual maximum
    % 	                        	end % if max_fir < 1
    %
    % 				%con = contourf(rot90(frmap),0:0.01:max(max(frmap)),'LineStyle','none');% plot the ratemap
    % 	                        pco = pcolor(frmap);			                                                        			% use the default (max firing rate = max rate map value) setting
    %                                 shading interp 		                                                                				% interpolate the rate map such that the pixels appeared to be smoothed
    %                                 set(pco,'linestyle','none');
    % 				caxis(cbar);
    %
    %                                 title(sprintf('SI %.2f b/s   SP %.2f%%   Cohe %.2f',skaggs, (spars*100), cohe),'FontSize', 8);
    %                                 set(gca, 'DataAspectRatio', [1 1 1]);
    %                                 set(gca, 'FontSize', 6);
    %
    %                                 axis off;
    %                                 p = get(gca, 'pos');				                                                			% p = colorbar parameters (x y (of bottom left corner) width height)
    %                                 cc = colorbar('EastOutside','FontSize', 8);                                                                  		% add colorbar to north of last plot
    % 				ylabel(cc, 'Hz');
    %                                 set(gca, 'pos', p);				                                                			% Adjusting values
    %                                 view(90,90);
    % %%
    %%%%%%%%%%%%%%% Plot: HD vs Speed Only when within directional range of
    %%%%%%%%%%%%%%% cell
    %
    %Get Head Direction
    
%     Vars=daVarsStruct;
%     in_polar1 = make_in_struct_for_rm (mtint,tetrode_1,cell_1,50,Vars.rm.smthKernPos,Vars.rm.binSizePosCm,Vars.rm.binSizeDir,0,Vars,'dir');
%     if size(in_polar1.posDir,2) ~= 1
%         in_polar1.posDir = in_polar1.posDir';
%     end
%     in = in_polar1;
%     
%     PolarMap_cell1 = GetRateMap (in);
%     circRm_cell1=PolarMap_cell1.map; %mean firing rate for each bin
%     HD_bin_centers = 3:6:357; %center of each bin
%     
%     
%     if(cell_type(plotnumber) <3)  %find the two peaks of the flip cell
%         
%         %USE PYJ'S METHOD TO FIND FLIP PEAKS
%         [~,sortIndex] = sort(circRm_cell1(:),'descend');
%         
%         peak1 = HD_bin_centers(sortIndex(1));
%         
%         %Circular correlation
%         circRm=circRm_cell1;    %to use Pierre's syntax
%         correlation = zeros(numel(circRm),2); %créer une 2ème colonne
%         for ibins = 0:numel(circRm)-1
%             %b=[];
%             b=circshift(circRm,ibins);%shifter de 1 bin pour pouvoir faire une corrélation en boucle
%             correlation(ibins+1,1) = corr(circRm,b,'type','pearson');%calculer la corrélation entre le vecteur circRm et le vecteur shifter
%             correlation(ibins+1,2) = ibins * Vars.rm.binSizeDir; %écrire dans la deuxième colonne les angles
%         end
%         %create a mask
%         binsize = 3 ;
%         ExtraC = [correlation(:,1)]';
%         mask = [];
%         %setup maxima & angle
%         maxima=[];
%         anglemaxima=[];
%         %loop to detect when there is a maxima
%         for iData = 1:numel(ExtraC)
%             if iData == 1
%                 mask=[ExtraC(end) ExtraC(iData:iData+1)];
%             elseif iData==numel(ExtraC)
%                 mask=[ExtraC(iData-1:iData) ExtraC(1)];
%             else
%                 mask=ExtraC(iData-1:iData+1);
%             end
%             if mask(2)==max(mask)
%                 maxima=[maxima; mask(2)];%store the maxima
%                 anglemaxima=[anglemaxima; iData];%store the angle maxima
%             end
%             mask = [];
%         end
%         
%         %find second peak in correlation
%         max2 = [];
%         if length(maxima)>=2
%             %reporter les 2ème pic
%             max2 = max(maxima(maxima~=max(maxima)));
%             %reporter l'angle correspondant au 2ème pic
%             angle_max2 = anglemaxima(maxima == max2)*Vars.rm.binSizeDir;
%             %ploter le 2ème peak max
%             angle_max2 = angle_max2(1,1);
%             
%             angle_max1 = 0;
%             %calculer l'angle entre les deux angles correspondant aux 2 corrélations maximales
%             distance_angle_peak = (abs(angle_max1-angle_max2)); %distance between two peaks
%             
%         else
%             max2 = []
%         end
%         
%         %Use peak distance to find second pfd of flip cell
%         bin_dist = round(distance_angle_peak/6); %num bins peak 2 is from peak1
%         
%         frst_poss_peak2 = sortIndex(1) + bin_dist;
%         
%         if frst_poss_peak2 > 60
%             frst_poss_peak2 = frst_poss_peak2 - 60;
%         end
%         
%         scnd_poss_peak2 = sortIndex(1) - bin_dist;
%         
%         if scnd_poss_peak2 < 0
%             scnd_poss_peak2 = scnd_poss_peak2 + 60;
%         end
%         
%         if(circRm_cell1(frst_poss_peak2) > circRm_cell1(scnd_poss_peak2))
%             peak2 = HD_bin_centers(frst_poss_peak2);
%         else
%             peak2 = HD_bin_centers(scnd_poss_peak2);
%         end
%         
%         
%     else        %FOR HD CELLS (celltpye == 3)
%         
%         [~,sortIndex] = sort(circRm_cell1(:),'descend');
%         
%         peak1 = HD_bin_centers(sortIndex(1));
%         peak2 = NaN;
%     end
    
    % %%
    % %%Putting the computed peaks on firing profile to check they're sensible...
    % figure()
    % plot([6:6:360;],circRm_cell1,'b','linewidth',2);
    % hold on
    %  xlim([1,360]);
    %  set(gca, 'XTick', [60:60:360]);
    %  %set(gca,'xticklabel',[60:60:360]);
    %  ylim([0,ceil(max(circRm_cell1))]);
    %  plot([peak1 peak1],[0 ceil(max(circRm_cell1))],'r--');
    %  if ~isnan(peak2)
    %      plot([peak2 peak2],[0 ceil(max(circRm_cell1))],'r--');
    %  end
    %
    %  newfilename = ['R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1)];
    %  cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/2linear_angular_speed/RSCFLIP/_peaks_detected');   %put into results folder
    %  saveas(gcf,newfilename,'jpg');
    %  close(gcf);
    %
    %  cd(savepath);
    %%
    %
    % %Bin occupancy and number of spikes in 2cm/s by 20 deg/s bins
    % %(they use 3 and 20, but I prefer 2cm/s - not important)
    %
    % %Speed
    % speed = mtint.pos.speed; %this is presumably speed in cm/s
    % %speed_binedges = [2:2:102];  %only binning speeds above 2cm/s
    % speed_binedges = [2:2:32];
    % num_speed_bins = numel(speed_binedges)-1;
    %
    %
    % peak1 = peak1 * (pi/180);
    % peak2 = peak2 * (pi/180);
    %HD
%     HeadDirection = in.posDir; %HD at 50ms sample rate
%     HeadDirection = HeadDirection * (pi/180); %lol radians
%     %
%     % %Getting current HD relative to PFD
%     % %what about when HD flips due to tracking error? Think about this a bit and
%     % %how to correct for it....
%     %
%     within_firing_range = false(numel(HeadDirection),1);
%     
%     if(cell_type(plotnumber)<3)  %just for RSC FLIP cells
%         for index = 1:numel(HeadDirection)
%             
%             distance1 = abs(atan2(sin(peak1-HeadDirection(index)),cos(peak1-HeadDirection(index))));
%             distance2 = (2*pi) - distance1;
%             
%             if(distance1<distance2)
%                 peak1dist = distance1;
%             else
%                 peak1dist = distance2;
%             end
%             
%             distance3 = abs(atan2(sin(peak2-HeadDirection(index)),cos(peak2-HeadDirection(index))));
%             distance4 = (2*pi) - distance3;
%             
%             if(distance3<distance4)
%                 peak2dist = distance3;
%             else
%                 peak2dist = distance4;
%             end
%             
%             if(peak1dist<(pi/4) || peak2dist<(pi/4))
%                 within_firing_range(index) = 1;
%             end
%         end
%         
%     else %if cell_type(plotnumber) == 3
%         for index = 1:numel(HeadDirection)      %just for RSC HD cells
%             
%             distance1 = abs(atan2(sin(peak1-HeadDirection(index)),cos(peak1-HeadDirection(index)))); %absolute PFD change from trial 1 to trial 5
%             distance2 = (2*pi) - distance1;
%             
%             if(distance1<distance2)
%                 distance = distance1;
%             else
%                 distance = distance2;
%             end
%             
%             if(distance<(pi/4))
%                 within_firing_range(index) = 1;
%             end
%         end
%     end
    %
    % MoveVarSpikes = zeros(num_speed_bins, 1); %array for the spikecounts
    % MoveVarDwell = zeros(num_speed_bins, 1); %array for dwelltime
    %
    % original_within_firing_range = within_firing_range;
    %
    % for speedindex = 1:num_speed_bins
    %
    %
    %         %find time bins where both variables are within bin range
    %         conjunctive_timebins = find(speed>=speed_binedges(speedindex) & speed<speed_binedges(speedindex+1)...
    %             & ~within_firing_range);
    %
    %         MoveVarDwell(speedindex) = numel(conjunctive_timebins)/ 50;   %number of 20ms time bins, so /50 to get time in seconds
    %
    %         for conjunctiveindex=1:numel(conjunctive_timebins)   %get time bins where both variables within binrange
    %             lower_timebin = conjunctive_timebins(conjunctiveindex) * (20/1000); %converting from bins, which are increments of 20ms, to seconds (units of SpikeTimes)
    %             upper_timebin = (conjunctive_timebins(conjunctiveindex) + 1) * (20/1000);
    %
    %             MoveVarSpikes(speedindex) = MoveVarSpikes(speedindex) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
    %         end
    % end
    %
    %
    % T1_MoveVarRates = MoveVarSpikes ./ MoveVarDwell;
    %
    % %correct for low dwell times
    % bin_exclusion_criterion = 0.5;  %0.5s% of the time
    % bin_exclusion_filter = MoveVarDwell<bin_exclusion_criterion;
    % T1_MoveVarRates(bin_exclusion_filter) = NaN;
    %
    % figure()
    % %now doing regression between firing rates and the bin mean
    % bin_centers = zeros(numel(T1_MoveVarRates),1);
    % for index = 1:numel(T1_MoveVarRates)
    %     bin_centers(index) = (speed_binedges(index) + speed_binedges(index+1))/2;
    % end
    % subplot(3,1,1);
    % %fit a regression line to scatter plot
    % x = bin_centers(~bin_exclusion_filter);                 %Speed is x
    % y = T1_MoveVarRates(~bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    % scatter(x,y,'.');
    % format long
    % X = [ones(length(x),1) x];
    % b = X\y;
    % yCalc = X*b;
    % hold on
    % plot(x,yCalc,'r');
    % xlabel('Running Speed (cm/s)');
    % ylabel('Firing Rate (Hz)');
    % R2 = 1 - sum((y - yCalc).^2)/sum((y - mean(y)).^2);
    %
    % running_speed_stats(plotnumber,1:3) = [b(2), b(1), R2]; %stats in the format slope intercept R^2
    %
    %
    % title([' Slope: ',num2str(b(2)),' Intercept: ',num2str(b(1)),' R^2', num2str(R2)]);
    %
    % %%
    % %%
    % %%%%%%%%%%%%%%% Plot: Firing Rate vs. AHV (as above plot - within PFD of cell)
    % %Get Head Direction
    %
    % Vars=daVarsStruct;
    % in_polar1 = make_in_struct_for_rm (mtint,tetrode_1,cell_1,50,Vars.rm.smthKernPos,Vars.rm.binSizePosCm,Vars.rm.binSizeDir,0,Vars,'dir');
    % if size(in_polar1.posDir,2) ~= 1
    %     in_polar1.posDir = in_polar1.posDir';
    % end
    % in = in_polar1;
    %
    % HeadDirection = in.posDir; %HD at 50ms sample rate
    % AngHeadVel = zeros(numel(HeadDirection),1);
    %
    %
    % for index = 1 : (numel(HeadDirection) - 1)
    %     distance = HeadDirection(index+1)-HeadDirection(index); %this works provided HeadDirection ranges 0-360
    %
    %     %correcting distance to be signed shortest distance between two HDs
    %     if(distance<-180)
    %         distance = distance + (360);
    %     elseif(distance>180)
    %         distance = distance - (360);
    %     end
    %
    %     AngHeadVel(index) = distance * 50; %multiplied by 50Hz to give deg/s
    % end
    %
    %
    % CorrectedAngHeadVel = AngHeadVel(AngHeadVel<500 & AngHeadVel>-500);
    % %%better to correct this by dwell time rather than artificially limiting?
    %
    % num_ahv_bins = 500; %number of bins for AHV data in range -500 to +500, 500 bins gives width of 2 deg/s per bin
    % [AHV_bincounts, AHV_binedges] = histcounts(CorrectedAngHeadVel,num_ahv_bins); %bin counts is number of observations (20ms time intervals) in each bin
    %
    % AHV_dwell = AHV_bincounts/50;  %time spent in AHV bin (divided by 50 as sample rate 50Hz) - could replace with pos_tb?
    %
    % %count spikes in each bin
    % AHV_spikecount = zeros(1,numel(AHV_bincounts));
    % for index = 1: numel(AHV_bincounts)
    %
    %     %get index of all time bins where AHV is within binrange, do for
    %     %non-corrected AHV to get the right index number to find spikes
    %
    %     AHV_timebins = find(AngHeadVel>=AHV_binedges(index) & AngHeadVel<AHV_binedges(index+1) & ~within_firing_range);
    %
    %     %now sum spikes that occur in those time bins
    %
    %     for jdx = 1: numel(AHV_timebins)
    %
    %         lower_timebin = AHV_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
    %         upper_timebin = (AHV_timebins(jdx) + 1) * (20/1000);
    %
    %         AHV_spikecount(index) = AHV_spikecount(index) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
    %
    %     end
    %
    % end
    %
    % AHV_spikecount = AHV_spikecount./AHV_dwell; %correct for amount of time spent at different speeds
    %
    % %get rid of undersampled AHV
    % total_AHV_dwell = numel(CorrectedAngHeadVel) * (20/1000);  %total time in ms (speed is sampled at 20ms intervals)
    % AHV_bin_exclusion_criterion = total_AHV_dwell/200;  %0.5% of the time
    % AHV_bin_exclusion_filter = AHV_dwell>AHV_bin_exclusion_criterion;
    %
    % %get bin centers
    % AHV_bin_centers = zeros(numel(AHV_spikecount),1);
    % for index = 1:numel(AHV_spikecount)
    %     AHV_bin_centers(index) = (AHV_binedges(index) + AHV_binedges(index+1))/2;
    % end
    %
    % subplot(3,1,2);
    % %fit a regression line to scatter plot
    % x = AHV_bin_centers(AHV_bin_exclusion_filter);                 %AHV is x
    % y = AHV_spikecount(AHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    % scatter(x,y,'.');
    % format long
    % X = [ones(length(x),1) x];
    % b = X\y';
    % yCalc = X*b;
    % hold on
    % plot(x,yCalc,'r');
    % xlabel('Head Rotation (deg/s)');
    % ylabel('Firing Rate (Hz)');
    % R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
    % [~,p_AHV] = vartest2(AHV_spikecount,yCalc);
    % title([' Slope: ',num2str(b(2)),' Intercept: ',num2str(b(1)),' R^2', num2str(R2),' p = ',num2str(p_AHV)]);
    %
    % AHV_stats(plotnumber,1:3) = [b(2), b(1), R2]; %stats in the format slope intercept R^2
    %
    %
    % %now do this with absolute AHV
    %
    % absAHV = abs(CorrectedAngHeadVel);
    %
    % num_absahv_bins = 250; %number of bins for AHV data in range 0 to +500, 250 bins gives width of 2 deg/s per bin
    % [absAHV_bincounts, absAHV_binedges] = histcounts(absAHV,num_absahv_bins); %bin counts is number of observations (20ms time intervals) in each bin
    %
    % absAHV_dwell = absAHV_bincounts/50;  %time spent in AHV bin (divided by 50 as sample rate 50Hz) - could replace with pos_tb?
    %
    % %put shortest distance to PFD through same filter as AHV and abs AHV
    % within_firing_range = within_firing_range(AngHeadVel<500 & AngHeadVel>-500);
    %
    % %count spikes in each bin
    % absAHV_spikecount = zeros(1,numel(absAHV_bincounts));
    % for index = 1: numel(absAHV_bincounts)
    %
    %     %get index of all time bins where AHV is within binrange, do for
    %     %non-corrected AHV to get the right index number to find spikes
    %
    %     absAHV_timebins = find(absAHV>=absAHV_binedges(index) & absAHV<absAHV_binedges(index+1) & ~within_firing_range);
    %
    %     %now sum spikes that occur in those time bins
    %
    %     for jdx = 1: numel(absAHV_timebins)
    %
    %         lower_timebin = absAHV_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
    %         upper_timebin = (absAHV_timebins(jdx) + 1) * (20/1000);
    %
    %         absAHV_spikecount(index) = absAHV_spikecount(index) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
    %
    %     end
    %
    % end
    %
    % absAHV_spikecount = absAHV_spikecount./absAHV_dwell; %correct for amount of time spent at different speeds
    %
    % %get rid of undersampled AHV
    % total_absAHV_dwell = numel(absAHV) * (20/1000);  %total time in ms (speed is sampled at 20ms intervals)
    % absAHV_bin_exclusion_criterion = total_absAHV_dwell/200;  %0.5% of the time
    % absAHV_bin_exclusion_filter = absAHV_dwell>absAHV_bin_exclusion_criterion;
    %
    % %get bin centers
    % absAHV_bin_centers = zeros(numel(absAHV_spikecount),1);
    % for index = 1:numel(absAHV_spikecount)
    %     absAHV_bin_centers(index) = (absAHV_binedges(index) + absAHV_binedges(index+1))/2;
    % end
    %
    %
    % subplot(3,1,3);
    % %fit a regression line to scatter plot
    % x = absAHV_bin_centers(absAHV_bin_exclusion_filter);                 %Abs AHV is x
    % y = absAHV_spikecount(absAHV_bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    % scatter(x,y,'.');
    % format long
    % X = [ones(length(x),1) x];
    % b = X\y';
    % yCalc = X*b;
    % hold on
    % plot(x,yCalc,'r');
    % xlabel('Head Rotation (deg/s)');
    % ylabel('Firing Rate (Hz)');
    % R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
    % [~,p_absAHV] = vartest2(absAHV_spikecount,yCalc);
    % title([' Slope: ',num2str(b(2)),' Intercept: ',num2str(b(1)),' R^2', num2str(R2),' p = ',num2str(p_absAHV)]);
    %
    % absAHV_stats(plotnumber,1:3) = [b(2), b(1), R2]; %stats in the format slope intercept R^2
    %
    % %setting overall figure title
    % set(gcf,'NextPlot','add');
    % axes;
    % h = title(['R',num2str(Rat),' ',DateString,'Tet',num2str(tetrode_1),'Cell',num2str(cell_1)]);
    % set(gca,'Visible','off');
    % set(h,'Visible','on');
    %
    % newfilename = ['R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1)];
    % cd(savepath);   %put into results folder
    % saveas(gcf,newfilename,'jpg');
    %
    % close(gcf);
    
    %%
    %%%%%%%%%%%%%%% Plot: Accelerations vs Spiking
    % %Linear acceleration
    % speed = mtint.pos.speed;
    % linear_acceleration = gradient(speed);
    %
    % % linacc_binedges = [2:2:100];  %only binning speeds above 4cm/s
    % [linacc_bincounts,linacc_binedges] = histcounts(linear_acceleration);
    %
    %
    % linacc_dwell = linacc_bincounts/50;  %time (in s) spent in acceleration bin (divided by 50 as sample rate 50Hz) - could replace with pos_tb variable?
    %
    % %count spikes in each bin
    % linacc_spikecount = zeros(1,numel(linacc_bincounts));
    % for index = 1: numel(linacc_bincounts)
    %
    %     %get index of all time bins where linear acceleration is within binrange
    %
    %     linacc_timebins = find(linear_acceleration>=linacc_binedges(index) & linear_acceleration<linacc_binedges(index+1));
    %
    %     %now sum spikes that occur in those time bins
    %
    %     for jdx = 1: numel(linacc_timebins)
    %
    %         lower_timebin = linacc_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
    %         upper_timebin = (linacc_timebins(jdx) + 1) * (20/1000);
    %
    %         linacc_spikecount(index) = linacc_spikecount(index) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
    %
    %     end
    %
    % end
    %
    % linacc_spikecount = linacc_spikecount./linacc_dwell; %correct for amount of time spent at different speeds
    %
    %
    %
    % %now doing regression between firing rates and the bin mean
    % bin_centers = zeros(numel(linacc_spikecount),1);
    % for index = 1:numel(linacc_spikecount)
    %     bin_centers(index) = (linacc_binedges(index) + linacc_binedges(index+1))/2;
    % end
    %
    % %need to discount bins where the rat doesn't spend >0.5% of the total time
    %
    % total_linacc_dwell = numel(linear_acceleration) * (20/1000);  %total time in ms (speed is sampled at 20ms intervals)
    % bin_exclusion_criterion = total_linacc_dwell/200;  %0.5% of the time
    % bin_exclusion_filter = linacc_dwell>bin_exclusion_criterion;
    %
    % figure()
    % subplot(2,1,1);
    % %fit a regression line to scatter plot
    % x = bin_centers(bin_exclusion_filter);                 %Speed is x
    % y = linacc_spikecount(bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    % scatter(x,y,'.');
    % format long
    % X = [ones(length(x),1) x];
    % b = X\y';
    % yCalc = X*b;
    % hold on
    % plot(x,yCalc,'r');
    % xlabel('Linear Acceleration (cm/s^2)');
    % ylabel('Firing Rate (Hz)');
    % R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
    % axis square;
    % title([' Slope: ',num2str(b(2)),' Intercept: ',num2str(b(1)),' R^2', num2str(R2)]);
    %
    %
    %
    % %%
    % %Angular acceleration now
    % angular_acceleration = gradient(AngHeadVel);
    %
    % %correct angular acceleration
    % angular_acceleration = angular_acceleration(AngHeadVel<500 & AngHeadVel>-500);
    %
    % [angacc_bincounts,angacc_binedges] = histcounts(angular_acceleration);
    %
    %
    % angacc_dwell = angacc_bincounts/50;  %time (in s) spent in acceleration bin (divided by 50 as sample rate 50Hz) - could replace with pos_tb variable?
    %
    % %count spikes in each bin
    % angacc_spikecount = zeros(1,numel(angacc_bincounts));
    % for index = 1: numel(angacc_bincounts)
    %
    %     %get index of all time bins where linear acceleration is within binrange
    %
    %     angacc_timebins = find(angular_acceleration>=angacc_binedges(index) & angular_acceleration<angacc_binedges(index+1));
    %
    %     %now sum spikes that occur in those time bins
    %
    %     for jdx = 1: numel(angacc_timebins)
    %
    %         lower_timebin = angacc_timebins(jdx) * (20/1000); %converting from AHV bins, which are increments of 20ms, to seconds (units of SpikeTimes)
    %         upper_timebin = (angacc_timebins(jdx) + 1) * (20/1000);
    %
    %         angacc_spikecount(index) = angacc_spikecount(index) + numel(SpikeTimes(SpikeTimes>=lower_timebin & SpikeTimes<upper_timebin));
    %
    %     end
    %
    % end
    %
    % angacc_spikecount = angacc_spikecount./angacc_dwell; %correct for amount of time spent at different speeds
    %
    %
    %
    % %now doing regression between firing rates and the bin mean
    % bin_centers = zeros(numel(angacc_spikecount),1);
    % for index = 1:numel(angacc_spikecount)
    %     bin_centers(index) = (angacc_binedges(index) + angacc_binedges(index+1))/2;
    % end
    %
    % %need to discount bins where the rat doesn't spend >0.5% of the total time
    %
    % total_angacc_dwell = numel(angular_acceleration) * (20/1000);  %total time in ms (speed is sampled at 20ms intervals)
    % bin_exclusion_criterion = total_angacc_dwell/200;  %0.5% of the time
    % bin_exclusion_filter = angacc_dwell>bin_exclusion_criterion;
    %
    % subplot(2,1,2);
    % %fit a regression line to scatter plot
    % x = bin_centers(bin_exclusion_filter);                 %Speed is x
    % y = angacc_spikecount(bin_exclusion_filter);            %Firing Rate is y, , put into same dimension as x
    % scatter(x,y,'.');
    % format long
    % X = [ones(length(x),1) x];
    % b = X\y';
    % yCalc = X*b;
    % hold on
    % plot(x,yCalc,'r');
    % xlabel('Angular Acceleration (cm/s^2)');
    % ylabel('Firing Rate (Hz)');
    % R2 = 1 - sum((y' - yCalc).^2)/sum((y' - mean(y)).^2);
    % axis square;
    % title([' Slope: ',num2str(b(2)),' Intercept: ',num2str(b(1)),' R^2', num2str(R2)]);
    %
    % newfilename = ['Accels_R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1)];
    % cd(savepath);   %put into results folder
    % saveas(gcf,newfilename,'jpg');
    %
    % close(gcf);
    
    %%
    %%%%%%%%%%%%%%% Recording %age of spikes fall outsie 45 deg either side of
    %%%%%%%%%%%%%%% PFD (both PFDs for flip cells)
    
    % SpikeTimesMS = (round(SpikeTimes*10000))/10;
    % SpikesTwentyMSbins = histcounts(SpikeTimesMS,0:20:(numel(original_within_firing_range)*20));
    %
    % SpikesNotWithinRange = SpikesTwentyMSbins(~original_within_firing_range);
    %
    % outsidePFDspikes(plotnumber) = (sum(SpikesNotWithinRange)/numel(SpikeTimes)) * 100; %proportion of spikes outside PFD
    % peaks(plotnumber,1) = peak1;
    % peaks(plotnumber,2) = peak2;
    %
    % outside_firing_range_spikes = false(numel(SpikeTimes),1); %index of
    % %spikes outside PFD
    % [~,whichbin] = histc(SpikeTimesMS,0:20:601000);
    % for index = 1:numel(SpikeTimes)
    %
    %     bin = whichbin(index);
    %
    %     if(~original_within_firing_range(bin))
    %         outside_firing_range_spikes(index) = 1;
    %     end
    % end
    %
    % figure();
    % plot(pox,poy,'k')
    %  				hold on
    %  				plot(spx(outside_firing_range_spikes),spy(outside_firing_range_spikes),'r.','MarkerSize',15)
    %
    %                                  axis([min(min(pox)) max(max(pox)) min(min(poy)) max(max(poy))]);
    %                                  set(gca, 'DataAspectRatio', [1 1 1]);
    %                                  set(gca, 'FontSize', 6);
    %                                  title(sprintf('Spikes: %.f',n_spikes),'FontSize', 8);
    %                                  axis off;
    %
    %
    
    %  num_HD_bins = 60;
    %  HDVarSpikes = zeros(num_HD_bins, 1); %array for the spikecounts
    %  HDVarDwell = zeros(num_HD_bins, 1); %array for dwelltime
    %
    %  HD_binedges = [0:6:360];
    %
    % for speedindex = 1:num_HD_bins
    %
    %
    %         %find time bins where both variables are within bin range
    %         conjunctive_timebins = find(HeadDirection>=HD_binedges(speedindex) & HeadDirection<HD_binedges(speedindex+1));
    %
    %         HDVarDwell(speedindex) = numel(conjunctive_timebins)/ 50;   %number of 20ms time bins, so /50 to get time in seconds
    %
    %         HDVarSpikes(speedindex) = sum(~original_within_firing_range(conjunctive_timebins));
    %
    % end
    %
    %
    % HDVarRates = HDVarSpikes ./ HDVarDwell;
    %
    %
    % figure()
    % plot(HDVarRates);
    
    %%NOW PLOT RATE VS HD NOT WITHIN RANGE
    
    %%
    %%%%%%%%%%%%%%% Calculating burst index, as per Valerio and Taube (2016)
    %
    % meanFR = numel(SpikeTimes)/600; %Firing rate in Hz over this trial
    % upper_limit = 1.75*meanFR;
    % lower_limit = 0.25*meanFR;
    %
    % %Bin firing rate in 1s bins
    %
    %  AllTime = [ 0  : 1 : 600]; %Pierre's recording time
    % [InstFirRate,~] = histcounts(SpikeTimes,AllTime) ; %actually just spike count in 1s bins
    %
    % burst_index(plotnumber) = (sum(InstFirRate>upper_limit) + sum(InstFirRate<lower_limit))/numel(InstFirRate);
    
    %%
    %%%%%%%%%%%%%%% ISIs for this region
    
    
    % %this_cell_ISI = diff((round(SpikeTimes*10000))/10); %ISIs in ms
    % this_cell_ISI = diff(SpikeTimes) * 1000;
    % if min(this_cell_ISI) < 1
    %     disp('What?');
    % end
    %
    % ISIpath = '/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/3ISI';
    % flip_path = '/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/3ISI/flip_histograms';
    % rschd_path = '/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/3ISI/rschd_histograms';
    % PoS_path = '/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/3ISI/PoS_histograms';
    % ADN_path = '/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/3ISI/ADN_histograms';
    %
    % if(flip_cells)
    %     %plot this cell's ISI histogram in the flip cell folder
    %     back = pwd;
    %     %cd(flip_path);
    %     cd(ADN_path);
    %     figure()
    %     TwoMsEdges = 0:2:10000;
    %     histogram(this_cell_ISI,TwoMsEdges);
    %     xlabel('ISI (ms)');
    %     ylabel('Count');
    %     title('ISI Histogram');
    %     flip_filename = ['ISIs_R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1)];
    %     saveas(gcf,flip_filename,'fig');
    %     close(gcf);
    %
    %     %Work out time to peak and time to half peak
    %
    %     [ISIbincounts,~] = histcounts(this_cell_ISI,TwoMsEdges); %bin counts
    %     [~,ISIsortIndex] = sort(ISIbincounts(:),'descend');
    %
    %     %ISIsortIndex(1) is the bin # where the max occurs
    %
    %      this_cell_max = TwoMsEdges(ISIsortIndex(1)) +1; %bin mean at maximum
    %
    %      %Only looking at the peak of ISI histogram to 1000ms after the peak
    %       x = 1:2:10000-1;      %bin centers
    %       x = x(ISIsortIndex(1):ISIsortIndex(1)+500)';              %transposed to column for fit
    %       y = ISIbincounts(ISIsortIndex(1):ISIsortIndex(1)+500)';   %transposed to column for it
    %
    %       %Fit exponential decay to ISI histogram from peak onwards
    %       f = fit(x,y,'exp1');
    %       coeffvals = coeffvalues(f);
    %
    %       fitted_line = coeffvals(1)*exp(coeffvals(2)*x); %exponential decay fitted
    %
    %       half_max_value = fitted_line(1)/2;
    %
    %       half_max_bin = find(fitted_line<half_max_value,1);%find where fitted_line reaches this value
    %
    %       %time in ms between peak and half_max is therefore 2*number of bins onwards
    %       %fit takes to reach half maximum value
    %
    %      ISI_peaks(plotnumber,1) = half_max_bin*2;
    %      ISI_peaks(plotnumber,2) = this_cell_max;
    %
    %     cd(back);
    %
    %     %ISIs for this cell added to record of ISIs for this cell's type
    %
    %     if(plotnumber == 1)
    %         %flip_ISIs = this_cell_ISI;
    %         ADN_ISIs = this_cell_ISI;
    %     else
    %         %flip_ISIs = cat(1,flip_ISIs,this_cell_ISI);  %stores all ISIs for rsc flip cells
    %         ADN_ISIs = cat(1,ADN_ISIs,this_cell_ISI);
    %     end
    % else
    %
    %     %plot this cell's ISI histogram in the rsc hd cell folder
    %
    %     back = pwd;
    %
    %     %cd(rschd_path);
    %     cd(PoS_path);
    %     figure()
    %     TwoMsEdges = 0:2:10000;
    %     histogram(this_cell_ISI,TwoMsEdges);
    %     xlabel('ISI (ms)');
    %     ylabel('Count');
    %     title('ISI Histogram');
    %     flip_filename = ['ISIs_R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1)];
    %     saveas(gcf,flip_filename,'fig');
    %     close(gcf);
    %
    %      %Work out time to peak and time to half peak
    %
    %     [ISIbincounts,~] = histcounts(this_cell_ISI,TwoMsEdges); %bin counts
    %     [~,ISIsortIndex] = sort(ISIbincounts(:),'descend');
    %
    %     %ISIsortIndex(1) is the bin # where the max occurs
    %
    %      this_cell_max = TwoMsEdges(ISIsortIndex(1)) +1; %bin mean at maximum
    %
    %      %Only looking at the peak of ISI histogram to 1000ms after the peak
    %       x = 1:2:10000-1;      %bin centers
    %       x = x(ISIsortIndex(1):ISIsortIndex(1)+500)';              %transposed to column for fit
    %       y = ISIbincounts(ISIsortIndex(1):ISIsortIndex(1)+500)';   %transposed to column for it
    %
    %       %Fit exponential decay to ISI histogram from peak onwards
    %       f = fit(x,y,'exp1');
    %       coeffvals = coeffvalues(f);
    %
    %       fitted_line = coeffvals(1)*exp(coeffvals(2)*x); %exponential decay fitted
    %
    %       %record exponential decay fit params
    %       ISI_exp_params(plotnumber,1) = coeffvals(1);
    %       ISI_exp_params(plotnumber,2) = coeffvals(2);
    %
    %       half_max_value = fitted_line(1)/2;
    %
    %       half_max_bin = find(fitted_line<half_max_value,1);%find where fitted_line reaches this value
    %
    %       %time in ms between peak and half_max is therefore 2*number of bins onwards
    %       %fit takes to reach half maximum value
    %
    %      ISI_peaks(plotnumber,1) = half_max_bin*2;
    %      ISI_peaks(plotnumber,2) = this_cell_max;
    %
    %      cd(back);
    %
    %     if(plotnumber == 1)
    %         %rschd_ISIs = this_cell_ISI;
    %         PoS_ISIs = this_cell_ISI;
    %     else
    %         %rschd_ISIs = cat(1,rschd_ISIs,this_cell_ISI); %stores all ISIs for rsc hd cells
    %         PoS_ISIs = cat(1,PoS_ISIs,this_cell_ISI);
    %     end
    % end
    %
    %
    %
    
    
%     %% CALCULATING ISIs ONLY WHEN HD IS WITHIN PFD RANGE OF CELL
%     
%     % This cell ISIs
%     this_cell_SpikeTimes = (round(SpikeTimes*10000))/10; %ISIs in ms
%     
%     % Get a list of times when HD enters and exits the range of the cell
%     within_firing_range(index);
%     transition_times = zeros(numel(within_firing_range,1)); %allocate too much memory
%     transition_count = 0;
%     for index = 1:numel(within_firing_range)-1
%         if(within_firing_range(index)~=within_firing_range(index+1))
%             transition_count = transition_count+1;
%             transition_times(transition_count) = index*20; %20ms bins, so this is transition time in ms.
%         end
%     end
%     
%     transition_times = transition_times(transition_times>0);
%     
%     % Need to change code depending on starting in the PFD of cell
%     % also need to take into account the length of transition_times
%     
%     odd_number = mod(numel(transition_times),2); %see if transition_times has an odd no of observations
%     
%     if(within_firing_range(1)) %if HD starts in PFD of cell
%         if(odd_number) %if odd number of transitions
%             move_in = transition_times(2:2:end-1);
%             move_out = transition_times(1:2:end);
%         else
%             move_in = transition_times(2:2:end);
%             move_out = transition_times(1:2:end-1);
%         end
%     else %if HD starts outside PFD of cell
%         if(odd_number)
%             move_in = transition_times(1:2:end);
%             move_out = transition_times(2:2:end-1);
%         else
%             move_in = transition_times(1:2:end-1);
%             move_out = transition_times(2:2:end);
%         end
%     end
%     
%     
%     %Now for each pair of in and out, take ISIs and add them to list of ISIs
%     %for this cell
%     
%     this_cell_PFD_ISI = zeros(1,1);
%     
%     if(within_firing_range(1)) %if head starts in PFD
%         if(move_out(end)>move_in(end)) %if head ends outside PFD
%             % take ISIs between move_in(n) and move_out(n+1)
%             for idx = 1:numel(move_in)
%                 tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes>=move_in(idx) & this_cell_SpikeTimes<=move_out(idx+1));
%                 if(numel(tempSTs>1))
%                     tempISI = diff(tempSTs);
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI); %take ISIs in this region
%                 end
%                 
%                 if(numel(tempSTs)) %if there's values
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,(tempSTs(1) - move_in(idx))); %take diff between first Spike in interval and start of PFD
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,move_out(idx+1) - tempSTs(end)); %take diff between end and last Spike in interval
%                 end
%             end
%             %add in ISIs between 0 and move_out(1)
%             tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes<=move_out(1));
%             if(numel(tempSTs>1))
%                 tempISI = diff(tempSTs);
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI);
%             end
%             if(numel(tempSTs)) %if there's values
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempSTs(1)); %take diff between first spike and start of simulation (= to first spike time)
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,move_out(1) - tempSTs(end)); %take diff between end and last Spike in interval
%             end
%         else %if head ends inside PFD
%             %take ISIs between move
%             for idx = 1:numel(move_out)-1 % if head ends inside PFD
%                 tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes>=move_in(idx) & this_cell_SpikeTimes<=move_out(idx+1));
%                 if(numel(tempSTs>1))
%                     tempISI = diff(tempSTs);
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI); %take ISIs in this region
%                 end
%                 if(numel(tempSTs)) %if there's values
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempSTs(1) - move_in(idx)); %take diff between first Spike in interval and start of PFD
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,move_out(idx+1) - tempSTs(end)); %take diff between end and last Spike in interval
%                 end
%             end
%             %add in ISIs for the last period inside the PFD of the cell
%             %(between move_in(end) and end of simulation)
%             tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes>=move_in(end));
%             if(numel(tempSTs>1))
%                 tempISI = diff(tempSTs);
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI); %take ISIs in this region
%             end
%             if(numel(tempSTs)) %if there's values
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempSTs(1) - move_in(end)); %take diff between first Spike in interval and start of PFD
%             end
%             
%         end
%         
%     else %if head starts outside PFD of the cell
%         if(move_out(end)>move_in(end)) %if head ends outside PFD
%             for idx = 1:numel(move_in)
%                 tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes>=move_in(idx) & this_cell_SpikeTimes<=move_out(idx));
%                 if(numel(tempSTs>1))
%                     tempISI = diff(tempSTs);
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI); %take ISIs in this region
%                 end
%                 if(numel(tempSTs)) %if there's values
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempSTs(1) - move_in(idx)); %take diff between first Spike in interval and start of PFD
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,move_out(idx) - tempSTs(end)); %take diff between end and last Spike in interval
%                 end
%             end
%         else %if head ends inside PFD
%             for idx = 1:numel(move_in)-1
%                 tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes>=move_in(idx) & this_cell_SpikeTimes<=move_out(idx));
%                 if(numel(tempSTs>1))
%                     tempISI = diff(tempSTs);
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI); %take ISIs in this region
%                 end
%                 if(numel(tempSTs)) %if there's values
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempSTs(1) - move_in(idx)); %take diff between first Spike in interval and start of PFD
%                     this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,move_out(idx) - tempSTs(end)); %take diff between end and last Spike in interval
%                 end
%             end
%             %Also take ISIs for the last time head is within PFD of cell
%             tempSTs = this_cell_SpikeTimes(this_cell_SpikeTimes>=move_in(end));
%             if(numel(tempSTs>1))
%                 tempISI = diff(tempSTs);
%             end
%             if(numel(tempSTs)) %if there's values
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempISI); %take ISIs in this region
%                 this_cell_PFD_ISI = cat(1,this_cell_PFD_ISI,tempSTs(1) - move_in(end)); %take diff between first Spike in interval and start of PFD
%             end
%         end
%         
%     end
%     
%     
%     switch cell_type(plotnumber)
%         case 1 %within-compartment flips
%             within_ISIs = cat(1,within_ISIs,this_cell_PFD_ISI);
%             within_spikecount = cat(1,within_spikecount,numel(this_cell_PFD_ISI));
%             
%             filename = ['R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1),'_PFD_ISI.mat'];
%             cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/PFD_ISI/within_flips');
%             save(filename, 'this_cell_PFD_ISI');
%             
%             
%             % FIT EXPONENTIAL
%             TwoMsEdges = 0:2:10000;
%             [ISIbincounts,~] = histcounts(this_cell_PFD_ISI,TwoMsEdges); %bin counts
%             [~,ISIsortIndex] = sort(ISIbincounts(:),'descend');
%             %ISIsortIndex(1) is the bin # where the max occurs
%             
%             this_cell_max = TwoMsEdges(ISIsortIndex(1)) +1; %bin mean at maximum
%             
%             x = 1:2:10000-1;      %bin centers
%             x = x(ISIsortIndex(1):ISIsortIndex(1)+500)';              %transposed to column for fit
%             y = ISIbincounts(ISIsortIndex(1):ISIsortIndex(1)+500)';   %transposed to column for it
%             
%             %Fit exponential decay to ISI histogram from peak onwards
%             f = fit(x,y,'exp1');
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
%             %Get fit params for this cell
%             
%             
%             within_peak = cat(1,within_peak,this_cell_max);
%             within_half_peak = cat(1,within_half_peak,half_max_bin*2);
%             within_constant_a = cat(1,within_constant_a,coeffvals(1));
%             within_constant_b = cat(1,within_constant_b,coeffvals(2));
%    
%             
%         case 2 %between-compartment flips
%             between_ISIs = cat(1,between_ISIs,this_cell_PFD_ISI);
%             between_spikecount = cat(1,between_spikecount,numel(this_cell_PFD_ISI));
%             
%             filename = ['R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1),'_PFD_ISI.mat'];
%             cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/PFD_ISI/between_flips');
%             save(filename, 'this_cell_PFD_ISI');
%             
%             % FIT EXPONENTIAL
%             TwoMsEdges = 0:2:10000;
%             [ISIbincounts,~] = histcounts(this_cell_PFD_ISI,TwoMsEdges); %bin counts
%             [~,ISIsortIndex] = sort(ISIbincounts(:),'descend');
%             %ISIsortIndex(1) is the bin # where the max occurs
%             
%             this_cell_max = TwoMsEdges(ISIsortIndex(1)) +1; %bin mean at maximum
%             
%             x = 1:2:10000-1;      %bin centers
%             x = x(ISIsortIndex(1):ISIsortIndex(1)+500)';              %transposed to column for fit
%             y = ISIbincounts(ISIsortIndex(1):ISIsortIndex(1)+500)';   %transposed to column for it
%             
%             %Fit exponential decay to ISI histogram from peak onwards
%             f = fit(x,y,'exp1');
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
%             %Get fit params for this cell
%             
%             
%             between_peak = cat(1,between_peak,this_cell_max);
%             between_half_peak = cat(1,between_half_peak,half_max_bin*2);
%             between_constant_a = cat(1,between_constant_a,coeffvals(1));
%             between_constant_b = cat(1,between_constant_b,coeffvals(2));
%             
% 
%         case 3 %RSC HD cells
%             RSCHD_ISIs = cat(1,RSCHD_ISIs,this_cell_PFD_ISI);
%             HD_spikecount = cat(1,HD_spikecount,numel(this_cell_PFD_ISI));
%             filename = ['R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1),'_PFD_ISI.mat'];
%             cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/PFD_ISI/RSC_HD');
%             save(filename, 'this_cell_PFD_ISI');
%             
%             % FIT EXPONENTIAL
%             TwoMsEdges = 0:2:10000;
%             [ISIbincounts,~] = histcounts(this_cell_PFD_ISI,TwoMsEdges); %bin counts
%             [~,ISIsortIndex] = sort(ISIbincounts(:),'descend');
%             %ISIsortIndex(1) is the bin # where the max occurs
%             
%             this_cell_max = TwoMsEdges(ISIsortIndex(1)) +1; %bin mean at maximum
%             
%             x = 1:2:10000-1;      %bin centers
%             x = x(ISIsortIndex(1):ISIsortIndex(1)+500)';              %transposed to column for fit
%             y = ISIbincounts(ISIsortIndex(1):ISIsortIndex(1)+500)';   %transposed to column for it
%             
%             %Fit exponential decay to ISI histogram from peak onwards
%             f = fit(x,y,'exp1');
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
%             %Get fit params for this cell
%             
%             
%             HD_peak = cat(1,HD_peak,this_cell_max);
%             HD_half_peak = cat(1,HD_half_peak,half_max_bin*2);
%             HD_constant_a = cat(1,HD_constant_a,coeffvals(1));
%             HD_constant_b = cat(1,HD_constant_b,coeffvals(2));
%             
%     end
%     
%     if(numel(this_cell_PFD_ISI)>145) %if cell passes the criterion for inclusion in analysis
%         [bin_counts_per_cell(plotnumber,:),~] = histcounts(this_cell_PFD_ISI,0:2:1000);
%     else
%         bin_counts_per_cell(plotnumber,:) = NaN;
%     end
    
    %%
    %%%%%%%%%%%%%%% Speed-Theta Analysis (codes from Giulio/Burgess)
    %
    % [ Stats] = ThetaAnalysisFW(mtint,1,1,'/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/1LFP_theta/to_delete',num2str(Rat),DateString,1);
    %
    %
    %
    % % LFP_frequency(plotnumber,1) = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.maxFreq;
    % % LFP_frequency(plotnumber,2) = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.bandMaxPower;
    %
    % frequencies = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.freqs;
    % powers = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.power;
    %
    %
    % delta_mask = frequencies>=1 & frequencies <=4;
    % theta_mask = frequencies>=7 & frequencies <=11;
    % delta_powers = powers(delta_mask);
    % mean_delta = mean(delta_powers);
    % theta_powers = powers(theta_mask);
    % mean_theta = mean(theta_powers);
    
    
    %theta_ratio(plotnumber,1) = mean_theta/mean_delta;
    %other version below:
    %alt_theta_ratio(plotnumber,1) = Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.bandMaxPower/mean_delta;
    
    
    % Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.freqs is the
    % frequencies
    
    %Stats.mtint.theta.PowerSpectrum.SpeedAllRangePowerSpectrum.power is the
    %power at these frequencies
    
    
    % speed_theta_stats(plotnumber,1) = Stats.mtint.theta.hilBeta; %slope of speed-theta
    % speed_theta_stats(plotnumber,2) =  Stats.mtint.theta.hilF0; %interecept of speed-theta
    % speed_theta_stats(plotnumber,3) =  Stats.mtint.theta.r_speed_theta_binned_data; %r value of speed-theta
    % speed_theta_stats(plotnumber,4) =  Stats.mtint.theta.p_speed_theta_binned_data; %p value of speed-theta
    
    
    %%
    %%%%%%%%%%%%%%% Intrinsic frequency analysis (codes from Giulio/Burgess)
    % Vars=daVarsStruct;
    % max_speed = Vars.pos.maxSpeedThetaBin;
    % NBinsOfSpikingThetaModulation = Vars.eeg.NBinsOfSpikingThetaModulation;
    % SpeedBinTheta    = (max_speed - 0) / NBinsOfSpikingThetaModulation ;%Vars.eeg.speedBinLengthCm;S
    % Ranges = 0 : SpeedBinTheta : max_speed;Ranges(find(Ranges<Vars.pos.minSpeed)) = [] ;
    % HalfBin = diff(Ranges) ;HalfBin=HalfBin(1)/2;
    % BinCentres = Ranges + HalfBin ;
    % BinCentres(find(BinCentres > max_speed)) = [] ;
    % SpeedRange={'All'};
    %
    % for iSpeed = 1 :   numel(BinCentres)
    % SpeedRange{iSpeed+1} = [num2str(BinCentres(iSpeed))] ;
    % end
    % clear iSpeed;
    %
    %
    % MinPossibleSpeed = 0;%Vars.pos.minSpeed;
    % speed = mtint.pos.speed ;
    % speed(find(speed < MinPossibleSpeed)) = NaN;
    % speed(find(speed > max_speed)) = NaN;
    % acBinSize=0.002;                                                                     in.acBinSize =acBinSize;
    % acWindow =0.5;                                                                       in.acWindow=acWindow;
    % thetaRange=Vars.eeg.psTh;                                                            in.thetaRange=thetaRange;
    % padToPow2=16;                                                                        in.padToPow2=padToPow2;
    % smthKernelWidth=Vars.eeg.psSmth(1);                                                  in.smthKernelWidth=smthKernelWidth;
    % smthKernelSigma=Vars.eeg.psSmth(2);                                                  in.smthKernelSigma=smthKernelSigma;
    % maxFreq=Vars.eeg.psMaxF;                                                             in.maxFreq=maxFreq;
    % s2nWidth = 2;                                                                        in.s2nWidth=s2nWidth;
    % ymax =[];                                                                            in.ymax=ymax;
    % PLOT_ON= 1;                                                                          in.PLOT_ON=PLOT_ON;
    % posMask = ones(length(speed) ,1);                                                    in.posMask=posMask;
    % in.spikeTimes=SpikeTimes;
    % %%%% Now running function
    % ret = intrinsic_freq_autoCorr(in);
    %
    % intrinsic_freq_stats(plotnumber,1) = ret.maxFreq;
    % intrinsic_freq_stats(plotnumber,2) = ret.bandMaxPower;
    %
    % frequencies = ret.freqs;
    % powers = ret.power;
    % delta_mask = frequencies>=1 & frequencies <=4;
    % theta_mask = frequencies>=7 & frequencies <=11;
    % delta_powers = powers(delta_mask);
    % mean_delta = mean(delta_powers);
    % theta_powers = powers(theta_mask);
    % mean_theta = mean(theta_powers);
    % intrinsic_freq_stats(plotnumber,3) = mean_theta/mean_delta;
    % intrinsic_freq_stats(plotnumber,4) = ret.bandMaxPower/mean_delta;
    %
    % %saving power spectrum
    % cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/1LFP_theta/intrinsic');
    % intrinsic_filename = ['R',num2str(Rat),'_',DateString,'_Tet',num2str(tetrode_1),'_Cell',num2str(cell_1)];
    % saveas(gcf,intrinsic_filename,'jpg');
    % close(gcf);
    % %%%%%%%%%%%%%%
    fclose('all');
    
    cd(startpath); %GO HOME
end

% cd(savepath);
% save('running_speed_stats.mat', 'running_speed_stats');
% save('AHV_stats.mat', 'AHV_stats');
% save('absAHV_stats.mat','absAHV_stats');
% save('outside_PFD.mat', 'outsidePFDspikes');
% save('calculated_peaks.mat','peaks');

% cd(ISIpath);
%
% if(flip_cells)
%     %save('ADN_ISIs.mat', 'ADN_ISIs');
%     save('ADN_ISI_peaks.mat', 'ISI_peaks');
%     save('ADN_ISI_exp_params.mat','ISI_exp_params');
%     %save('flip_ISI_peaks.mat', 'ISI_peaks');
%     %save('flip_ISI_exp_params.mat','ISI_exp_params');
% else
%     %save('PoS_ISIs.mat', 'PoS_ISIs');
%     save('PoS_ISI_peaks.mat', 'ISI_peaks');
%     save('PoS_ISI_exp_params.mat','ISI_exp_params');
%     %save('rschd_ISI_peaks.mat', 'ISI_peaks');
%     %save('rschdp_ISI_exp_params.mat','ISI_exp_params');
% end

%cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/1LFP_theta');
% save('speed_theta_stats.mat','speed_theta_stats');
% save('LFP_frequency.mat','LFP_frequency');
%save('theta_ratio_all.mat','theta_ratio');
%save('theta_alt_ratio_all.mat','alt_theta_ratio');
% cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/1LFP_theta/intrinsic');
%save('corrected_intrinsic_freq_stats.mat', 'intrinsic_freq_stats');

%cd('/Volumes/KJ Filestore/Jeffery lab/Hector Page/flip cell work/movement_correlates/PFD_ISI');

% %saving overall ISIs
% save('within_ISIs.mat', 'within_ISIs');
% save('between_ISIs.mat', 'between_ISIs');
% save('RSCHD_ISIs.mat', 'RSCHD_ISIs');
% 
% %saving bin counts on a per-cell basis
% save('bin_counts_per_cell.mat','bin_counts_per_cell');
% 
% % saving fit parameters
% 
% save('within_peak.mat','within_peak');
% save('within_half_peak.mat','within_half_peak');
% save('within_constant_a.mat','within_constant_a');
% save('within_constant_b.mat','within_constant_b');
% save('within_spikecount.mat','within_spikecount');
% 
% save('between_peak.mat','between_peak');
% save('between_half_peak.mat','between_half_peak');
% save('between_constant_a.mat','between_constant_a');
% save('between_constant_b.mat','between_constant_b');
% save('between_spikecount.mat','between_spikecount');
% 
% save('HD_peak.mat','HD_peak');
% save('HD_half_peak.mat','HD_half_peak');
% save('HD_constant_a.mat','HD_constant_a');
% save('HD_constant_b.mat','HD_constant_b');
% save('HD_spikecount.mat','HD_spikecount');

cd(startpath);
end











