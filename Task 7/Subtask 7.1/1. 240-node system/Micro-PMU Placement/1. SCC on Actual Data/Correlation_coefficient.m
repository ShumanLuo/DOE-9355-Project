%% Spearman's Correlation Coefficient
% 240-node System
clc
clear
close all

load voltage_names
load  Correlation_True_voltages

%%
Vbase1 = 13800/sqrt(3);
% R_input is come from MATLAB_OpenDSS_lowa 
R_input_voltage = R_input; % in the loaded mat file above there is only voltage
%Convert voltages to p.u.
R_input_voltage = [R_input_voltage(:,[1:size(R_input_voltage,2)/2])/Vbase1,R_input_voltage(:,[size(R_input_voltage,2)/2+1:end])];
%Remove columns related to source bus and feeder head
indx = [1:6];
R_input_voltage(:,[indx,indx+size(R_input_voltage,2)/2]) = [];
%Remove the first 6 nodes related to source bus and feeder head
node_voltage_names_separated = node_voltage_names_separated(7:end,:);
%% Order voltages
%Here we sort arrays in "node_voltage_names_separated". To make
%R_input_voltage consistent with this ordering, we also get sort indexes, to use them for ordering R_input_voltage later on 
[node_voltage_names_separated_ordered, sort_index] = sortrows(node_voltage_names_separated, 1);
%% Getting indexes of all phases separately
%These are indexes related to all feeders
% Getting indexes for voltage magnitude
phase_A_voltage_mag_index_whole= [];
phase_B_voltage_mag_index_whole= [];
phase_C_voltage_mag_index_whole= [];
for i = 1:size (node_voltage_names_separated_ordered,1)
    if strcmp(node_voltage_names_separated_ordered{i,2},'1')
        phase_A_voltage_mag_index_whole = [phase_A_voltage_mag_index_whole,i];
    elseif strcmp(node_voltage_names_separated_ordered{i,2},'2')
        phase_B_voltage_mag_index_whole = [phase_B_voltage_mag_index_whole,i];
    elseif strcmp(node_voltage_names_separated_ordered{i,2},'3')
        phase_C_voltage_mag_index_whole = [phase_C_voltage_mag_index_whole,i];
    end
end
% Sort R_input_voltage based on sort indexes
R_input_voltage = R_input_voltage(:,[sort_index,sort_index+size(R_input_voltage,2)/2]);
% Getting indexes for voltage angles
phase_A_voltage_ang_index_whole = phase_A_voltage_mag_index_whole + size(R_input_voltage,2)/2;
phase_B_voltage_ang_index_whole = phase_B_voltage_mag_index_whole + size(R_input_voltage,2)/2;
phase_C_voltage_ang_index_whole = phase_C_voltage_mag_index_whole + size(R_input_voltage,2)/2;

%% In case you need the heatmaps for all feeders in the same plot
R_mag_A_whole = corr([R_input_voltage(:,phase_A_voltage_mag_index_whole)],'type','Spearman');
R_mag_B_whole = corr([R_input_voltage(:,phase_B_voltage_mag_index_whole)],'type','Spearman');
R_mag_C_whole = corr([R_input_voltage(:,phase_C_voltage_mag_index_whole)],'type','Spearman');
R_ang_A_whole = corr([R_input_voltage(:,phase_A_voltage_ang_index_whole)],'type','Spearman');
R_ang_B_whole = corr([R_input_voltage(:,phase_B_voltage_ang_index_whole)],'type','Spearman');
R_ang_C_whole = corr([R_input_voltage(:,phase_C_voltage_ang_index_whole)],'type','Spearman');
heatmap(R_ang_A_whole)
%% Let's separate feeders
feeder_A_order = [];
feeder_B_order = [];
feeder_C_order = [];
for i = 1:size(node_voltage_names_separated_ordered,1)
    if node_voltage_names_separated_ordered{i}(4) == '1'
        feeder_A_order = [feeder_A_order, i];
    elseif node_voltage_names_separated_ordered{i}(4) == '2'
        feeder_B_order = [feeder_B_order, i];
    else
        feeder_C_order = [feeder_C_order, i];
    end
end

% Getting indexes separately for 3 feeders
%Find phase indexes in all feeders separately
%Magnitude indexes
feeder_A_phase_A_mag_indx = intersect (feeder_A_order,phase_A_voltage_mag_index_whole);
feeder_A_phase_B_mag_indx = intersect (feeder_A_order,phase_B_voltage_mag_index_whole);
feeder_A_phase_C_mag_indx = intersect (feeder_A_order,phase_C_voltage_mag_index_whole);

feeder_B_phase_A_mag_indx = intersect (feeder_B_order,phase_A_voltage_mag_index_whole);
feeder_B_phase_B_mag_indx = intersect (feeder_B_order,phase_B_voltage_mag_index_whole);
feeder_B_phase_C_mag_indx = intersect (feeder_B_order,phase_C_voltage_mag_index_whole);

feeder_C_phase_A_mag_indx = intersect (feeder_C_order,phase_A_voltage_mag_index_whole);
feeder_C_phase_B_mag_indx = intersect (feeder_C_order,phase_B_voltage_mag_index_whole);
feeder_C_phase_C_mag_indx = intersect (feeder_C_order,phase_C_voltage_mag_index_whole);

%Phase Angle indexes
feeder_A_phase_A_ang_indx = feeder_A_phase_A_mag_indx + +size(R_input_voltage,2)/2;
feeder_A_phase_B_ang_indx = feeder_A_phase_B_mag_indx + +size(R_input_voltage,2)/2;
feeder_A_phase_C_ang_indx = feeder_A_phase_C_mag_indx + +size(R_input_voltage,2)/2;

feeder_B_phase_A_ang_indx = feeder_B_phase_A_mag_indx + +size(R_input_voltage,2)/2;
feeder_B_phase_B_ang_indx = feeder_B_phase_B_mag_indx + +size(R_input_voltage,2)/2;
feeder_B_phase_C_ang_indx = feeder_B_phase_C_mag_indx + +size(R_input_voltage,2)/2;

feeder_C_phase_A_ang_indx = feeder_C_phase_A_mag_indx + +size(R_input_voltage,2)/2;
feeder_C_phase_B_ang_indx = feeder_C_phase_B_mag_indx + +size(R_input_voltage,2)/2;
feeder_C_phase_C_ang_indx = feeder_C_phase_C_mag_indx + +size(R_input_voltage,2)/2;

feeders_index{1} = [feeder_A_order,feeder_A_order+size(R_input_voltage,2)/2];
feeders_index{2} = [feeder_B_order,feeder_B_order+size(R_input_voltage,2)/2];
feeders_index{3} = [feeder_C_order,feeder_C_order+size(R_input_voltage,2)/2];

%% Heatmaps
%Change the index insideR_input_voltage to get the heatmaps of all feeders
%and phases separately
heatmap(corr([R_input_voltage(:,feeder_A_phase_A_mag_indx)],'type','Spearman'))


%% Next Step:
% 1- plot heatmaps based on noisy data (to see better results, reduce the voltage level noise)
% 2- 
