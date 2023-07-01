clc
clear
close all
%% voltages and currents
%Load voltage and current values & names created in MATLAB_OPENDSS-24node.m file
%Voltage magnitudes in p.u and angles in radian, considering 3 PVs
load 1000Scen_3PVs_with_current_pu_860_848_mid822_varing_feeder_head

%% Import R_input_voltage_fixed_Reg_taps (Run only for do a study)

R_input_voltage_fixed_Reg_taps = readmatrix('R_input_voltage_fixed_Reg_taps.csv');
R_input_voltage = R_input_voltage_fixed_Reg_taps;
%% removing constant voltage magnitudes and their corresponding phase angles.
temp1  = std(R_input_voltage(:,1:size(R_input_voltage,2)/2));
temp2 = find (temp1 <0.001); 
node_voltage_names_separated(temp2,:) = [];
temp2 = [temp2,temp2+size(R_input_voltage,2)/2];
R_input_voltage(:,temp2) = [];

%% voltage - voltage analysis
%% finding the index voltage for each phase
phase_A_voltage_mag_index= [];
phase_B_voltage_mag_index= [];
phase_C_voltage_mag_index= [];
for i = 1:size (node_voltage_names_separated,1)
    if strcmp(node_voltage_names_separated{i,2},'1')
        phase_A_voltage_mag_index = [phase_A_voltage_mag_index,i];
    elseif strcmp(node_voltage_names_separated{i,2},'2')
        phase_B_voltage_mag_index = [phase_B_voltage_mag_index,i];
    elseif strcmp(node_voltage_names_separated{i,2},'3')
        phase_C_voltage_mag_index = [phase_C_voltage_mag_index,i];
    end
end

%Reordeing some indexes, because they're not in the correct ordring based
%on radial poser system diagarm (we need indexes in down-stream manner)
phase_A_voltage_mag_index = [1,4,7,10,14,17,20,23,26,30,31,27,33,36,39,70,42,73,84,45,76,48,51,60,63,66,77,54,57,80];
phase_B_voltage_mag_index = [2,5,8,11,13,15,18,21,24,28,32,34,37,40,69,71,43,74,85,46,49,52,61,64,67,78,55,58,81,83];
phase_C_voltage_mag_index = [3,6,9,12,16,19,22,25,29,35,38,41,72,44,75,86,47,50,53,62,65,68,79,56,59,82];

% In case you are doing that study >>> Fixed Reg Taps
% phase_A_voltage_mag_index = [17,20,23,26,30,31,27,33,36,39,70,42,73,84,45,76,48,51,60,63,66,77,54,57,80]-14;
% phase_B_voltage_mag_index = [15,18,21,24,28,32,34,37,40,69,71,43,74,85,46,49,52,61,64,67,78,55,58,81,83]-14;
% phase_C_voltage_mag_index = [19,22,25,29,35,38,41,72,44,75,86,47,50,53,62,65,68,79,56,59,82]-14;
phase_A_voltage_mag_index = [4,7,10,14,17,20,23,26,30,31,27,33,36,39,70,42,73,84,45,76,48,51,60,63,66,77,54,57,80]-3;
phase_B_voltage_mag_index = [5,8,11,13,15,18,21,24,28,32,34,37,40,69,71,43,74,85,46,49,52,61,64,67,78,55,58,81,83]-3;
phase_C_voltage_mag_index = [6,9,12,16,19,22,25,29,35,38,41,72,44,75,86,47,50,53,62,65,68,79,56,59,82]-3;



phase_A_voltage_ang_index = phase_A_voltage_mag_index + size(R_input_voltage,2)/2;
phase_B_voltage_ang_index = phase_B_voltage_mag_index + size(R_input_voltage,2)/2;
phase_C_voltage_ang_index = phase_C_voltage_mag_index + size(R_input_voltage,2)/2;

phase_A_voltage_names = node_voltage_names_separated(phase_A_voltage_mag_index ,1);
phase_B_voltage_names = node_voltage_names_separated(phase_B_voltage_mag_index ,1);
phase_C_voltage_names = node_voltage_names_separated(phase_C_voltage_mag_index ,1);


R_mag_A = corr(R_input_voltage(:,phase_A_voltage_mag_index),'type','Spearman');
R_mag_B = corr(R_input_voltage(:,phase_B_voltage_mag_index),'type','Spearman');
R_mag_C = corr(R_input_voltage(:,phase_C_voltage_mag_index),'type','Spearman');
R_ang_A = corr(R_input_voltage(:,phase_A_voltage_ang_index),'type','Spearman');
R_ang_B = corr(R_input_voltage(:,phase_B_voltage_ang_index),'type','Spearman');
R_ang_C = corr(R_input_voltage(:,phase_C_voltage_ang_index),'type','Spearman');
min_correlation = [min(min(R_mag_A)),min(min(R_mag_B)),min(min(R_mag_C)),min(min(R_ang_A)),min(min(R_ang_B)),min(min(R_ang_C))];

%% Plotting heatmaps - Magnitudes
figure
heatmap(R_mag_A,'FontName', 'Times New Roman', 'FontSize', 10,'Colormap',parula)
ax = gca;
ax.XData =[phase_A_voltage_names];
ax.YData =[phase_A_voltage_names];
title ('Phase A voltage magnitude CC matrix')
xlabel('node voltage name')
ylabel('node voltage name')
figure
heatmap(R_mag_B,'FontName', 'Times New Roman', 'FontSize', 10,'Colormap',parula)
ax = gca;
ax.XData =[phase_B_voltage_names];
ax.YData =[phase_B_voltage_names];
title ('Phase B voltage magnitude CC matrix')
xlabel('node voltage name')
ylabel('node voltage name')
figure
heatmap(R_mag_C,'FontName', 'Times New Roman', 'FontSize', 10,'Colormap',parula)
ax = gca;
ax.XData =[phase_C_voltage_names];
ax.YData =[phase_C_voltage_names];
title ('Phase C voltage magnitude CC matrix')
xlabel('node voltage name')
ylabel('node voltage name')

%% Plotting heatmaps - Angles
figure
heatmap(R_ang_A,'FontName', 'Times New Roman', 'FontSize', 10,'Colormap',parula)
ax = gca;
ax.XData =[phase_A_voltage_names];
ax.YData =[phase_A_voltage_names];
title ('Phase A voltage phase angle CC matrix')
xlabel('node voltage name')
ylabel('node voltage name')
figure
heatmap(R_ang_B,'FontName', 'Times New Roman', 'FontSize', 10,'Colormap',parula)
ax = gca;
ax.XData =[phase_B_voltage_names];
ax.YData =[phase_B_voltage_names];
title ('Phase B voltage phase angle CC matrix')
xlabel('node voltage name')
ylabel('node voltage name')
figure
heatmap(R_ang_C,'FontName', 'Times New Roman', 'FontSize', 10,'Colormap',parula)
ax = gca;
ax.XData =[phase_C_voltage_names];
ax.YData =[phase_C_voltage_names];
title ('Phase C voltage phase angle CC matrix')
xlabel('node voltage name')
ylabel('node voltage name')



