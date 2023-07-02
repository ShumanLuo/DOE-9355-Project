% This code is designed to extract the branch data of four different test
% systems
clear all;
close all;
clc;

% Choose the system and load it
% casename = 'case30';
% casename = 'case118';
% casename = 'case300';
% casename = 'case2383wp';
casename = 'case_ACTIVSg2000'

mpc = loadcase(casename);

% This is only for case300 which has awkward bus numbers
if strcmp(casename, 'case300') || strcmp(casename, 'case_ACTIVSg2000')
    bus_mat = mpc.bus;
    branch_mat = mpc.branch;
    for i=1:size(branch_mat,1)
    from_indx = branch_mat(i,1);
    new_from_indx = find(bus_mat(:,1)==from_indx);
    branch_mat(i,1) = new_from_indx;
    
    to_indx = branch_mat(i,2);
    new_to_indx = find(bus_mat(:,1)==to_indx);
    branch_mat(i,2) = new_to_indx;
    end
    mpc.branch = branch_mat;
end

% Get the branch info
A = mpc.branch(:,[1:5,9]);
A(:,end) = A(:,end) + (A(:,end) == 0) * (1);

% Store branch info
filename = sprintf('bdata_%s.csv', casename);
writematrix(A, filename)
