clc
clear all
close all
%% Three-Phase Linear State Estimate for Iowa system:
tic;

NodeData = xlsread('Iowa_Node_Data.xlsx'); % Have the node data sorted based upon the bus numbers 
LineData = xlsread('Iowa_Line_Data.xlsx','Line_Data_Final');
% load H_Iowa.mat;
load H_Iowa_multiple_pmu_same_bus.mat
% load data_6000_GMM.mat;

% load data_2500_GMM_no_SMerror_fixed_Instrument_error.mat

%% Latest Data:
% load data_2500_GMM_no_SMerror_Recent_Instrument_error.mat;
load data_2500_NoSMerror_NoInstrument_TVE.mat


Vm_Ar = VoltBusNameToBusNumber(node_voltage_names_separated);
% Vm_Ar = horzcat(Vm_Ar,voltage_actual_full_ready); % uncomment if you want actual data in z vector
Vm_Ar = horzcat(Vm_Ar,voltage_error_full_ready); % uncomment if you want noisy (1% TVE: Gaussina) data in z vector
Vm_Ar = sortrows(Vm_Ar,2);
Vm_Ar = sortrows(Vm_Ar,1);

Vm_Tr_Ar  = VoltBusNameToBusNumber(node_voltage_names_separated);
Vm_Tr_Ar = horzcat(Vm_Tr_Ar,voltage_actual_full_ready);
Vm_Tr_Ar = sortrows(Vm_Tr_Ar,2);
Vm_Tr_Ar = sortrows(Vm_Tr_Ar,1);

Im_Ar = CurrentBusNameToBusNumber(All_currents_WLS_names_ready);
% Im_Ar = horzcat(Im_Ar,All_currents_WLS_actual_ready_to_send); % uncomment if you want actual data in z vector
Im_Ar = horzcat(Im_Ar,All_currents_WLS_ready_to_send); % uncomment if you want noisy (1% TVE: Gaussina) data in z vector

time_read = toc;

tic;
[row_Vm_Ar,col_Vm_Ar] = size(Vm_Ar);
NoOfScene = (col_Vm_Ar-2)/2;
NoOfState = row_Vm_Ar;

Sum_mag_err_sc = 0;
Sum_ang_err_sc = 0;
Sum_mag_err_per_sc = 0;

BusMap = 1:length(NodeData);
NoOfNode = length(NodeData);

Sbase = 100*10^6;
Vbase1 = 13.8*10^3;
Ibase1 = Sbase/(sqrt(3)*Vbase1);
Zbase1 = (Vbase1^2)/Sbase;

PhInNode = xlsread('PhaseInNode_Iowa.xlsx');
PhPerNode = [];
for i = 1:length(BusMap)
    bus = BusMap(i);
    if PhInNode(i,2)~=0 && PhInNode(i,3)~=0 && PhInNode(i,4)~=0
        PhPerNode(i,1) = 3;
    elseif (PhInNode(i,2)==0 && PhInNode(i,3)~=0 && PhInNode(i,4)~=0) || (PhInNode(i,2)~=0 && PhInNode(i,3)==0 && PhInNode(i,4)~=0) || (PhInNode(i,2)~=0 && PhInNode(i,3)~=0 && PhInNode(i,4)==0)
        PhPerNode(i,1) = 2;
    elseif (PhInNode(i,2)~=0 && PhInNode(i,3)==0 && PhInNode(i,4)==0) || (PhInNode(i,2)==0 && PhInNode(i,3)~=0 && PhInNode(i,4)==0) || (PhInNode(i,2)==0 && PhInNode(i,3)==0 && PhInNode(i,4)~=0)
        PhPerNode(i,1) = 1;
    end     
end 
% uPMUloc_all = xlsread('PMU_Loc_Iowa.xlsx');
uPMUloc_all = xlsread('PMU_Loc_Iowa.xlsx','Multiple_PMU_Same_Bus');
uPMUloc = uPMUloc_all(:,1);
LineMonitor = xlsread('PMU_Line_Iowa.xlsx');


for nscene = 1:NoOfScene
    
Vm = [];
Vm(:,1) = Vm_Ar(:,1);
Vm(:,2) = Vm_Ar(:,2);
Vm(:,3) = Vm_Ar(:,2*nscene+1);
Vm(:,4) = Vm_Ar(:,2*nscene+2);
   
Vm_Tr = [];
Vm_Tr(:,1) = Vm_Tr_Ar(:,1);
Vm_Tr(:,2) = Vm_Tr_Ar(:,2);
Vm_Tr(:,3) = Vm_Tr_Ar(:,2*nscene+1);
Vm_Tr(:,4) = Vm_Tr_Ar(:,2*nscene+2);
   
Im = [];
Im(:,1) = Im_Ar(:,1);
Im(:,2) = Im_Ar(:,2);
Im(:,3) = Im_Ar(:,3);
Im(:,4) = Im_Ar(:,2*nscene+2);
Im(:,5) = Im_Ar(:,2*nscene+3);


% "TotMeas" is the total number of voltage measurements:
% "sum(PhPerNode)" is the total number of power system states:
% Dimension of H_volt is "TotMeas"x"sum(PhPerNode)"
TotMeas = 0;
for i = 1:length(uPMUloc)
    TotMeas = TotMeas + PhPerNode(uPMUloc(i));    
end

row = 1;
V_meas = [];

for i = 1:length(uPMUloc)
    bus = uPMUloc(i);    
    if (bus==1)
        Num_Ph_Bus = 0;
    else
        Num_Ph_Bus = sum(PhPerNode([1:bus-1]));
    end
    for j = 1:PhPerNode(bus)
        H_V(row,Num_Ph_Bus+j) = 1;
        H_V_r(row,Num_Ph_Bus+j) = 1;
        H_V_i(row,Num_Ph_Bus+j) = 1;
        row = row + 1;
    end
    PhInNode_sel = [];np = 1;
    
    for k = 2:4
        if (PhInNode(bus,k)~=0)
            PhInNode_sel(np) = k-1; 
            np = np+1;
        end
    end
        
    % Get voltage measurements from new format:
    bus_org = NodeData(bus);
    V_temp = [];countV = 1;
    for p = 1:length(PhInNode_sel)
        ph = PhInNode_sel(p);
        for z = 1:length(Vm(:,1))
            if (bus_org==Vm(z,1)) && (ph==Vm(z,2))
                Mag = Vm(z,3);
                Ang = Vm(z,4);
                Volt_r = Mag*cosd(Ang);
                Volt_i = Mag*sind(Ang);
                V_temp(countV,1) = complex(Volt_r,Volt_i);
                countV = countV + 1;
                break;
            end
        end
    end
    if (PhPerNode(bus)~=length(V_temp))
        
    end
    
     V_meas = vertcat(V_meas,V_temp);
end

%% Find H_current:
count = 1;
count_split = 1;
TotalPhase = sum(PhPerNode);
H_I = zeros(1,TotalPhase);

for i = 1:length(LineMonitor(:,1)) 
    Bus1 = LineMonitor(i,1);
    Bus2 = LineMonitor(i,2);
    PhInLine = []; countPh = 1;
    for c = 2:4
        if (PhInNode(Bus1,c)~=0 && PhInNode(Bus2,c)~=0)
            PhInLine(countPh) = c-1;
            countPh = countPh+1;
        end            
    end        
    num_ph = length(PhInLine);
    % Get the current measurements:
    Bus1_org = NodeData(Bus1);
    Bus2_org = NodeData(Bus2);
    Curr_Fbus = [];count_I_F = 1;
    for p = 1:length(PhInLine)
        ph = PhInLine(p);
        for z = 1:length(Im(:,1))
            if ((Bus1_org==Im(z,1)) && (Bus2_org==Im(z,2)) && (ph==Im(z,3)))                
                Mag = Im(z,4)/Ibase1;                
                Ang = Im(z,5);
                Curr_r = Mag*cosd(Ang);
                Curr_i = Mag*sind(Ang);
                Curr_Fbus(count_I_F,1) = complex(Curr_r,Curr_i);
                count_I_F = count_I_F + 1;
                break;            
            end
        end
    end   
    I_meas([count:count+num_ph-1],1) = Curr_Fbus;    
    count = count+num_ph;       
end

%% Concatenate all measurements:
Meas = vertcat(V_meas,I_meas);
H_pseudo = ((H'*H)^-1)*(H');
x = H_pseudo*Meas;


%% Find the xtrue:
% Assumes that the Vm matrix is sorted based upon the bus numbers in the
% first column
xtr = [];
[row_Vm, col_Vm] = size(Vm_Tr);
count = 1;
for r = 1:row_Vm
      V_mag = Vm_Tr(r,3);
      V_ang = Vm_Tr(r,4);
      V_r = V_mag*cosd(V_ang);
      V_i = V_mag*sind(V_ang);
      xtr(count,1) = complex(V_r,V_i);
      count = count + 1;      
end
Meas_p = H*xtr;
x_mag = abs(x);
xtr_mag = abs(xtr);
diff_mag = abs(xtr_mag-x_mag);
% sum_diff_mag = sum(diff_mag);
% Sum_mag_err_sc = Sum_mag_err_sc + sum_diff_mag;
diff_mag_per = diff_mag./xtr_mag; 
sum_diff_mag_per = sum(diff_mag_per);
n = length(diff_mag_per);
MAPE_mag_sc = sum_diff_mag_per/n;
Sum_mag_err_per_sc = Sum_mag_err_per_sc + sum_diff_mag_per;

x_ang = angle(x)*180/pi;
xtr_ang = angle(xtr)*180/pi;
diff_ang = abs(xtr_ang-x_ang);
sum_diff_ang = sum(diff_ang);
n = length(diff_ang);
MAE_ang = sum_diff_ang/n;
Sum_ang_err_sc = Sum_ang_err_sc + sum_diff_ang;

diff_ang_ar(:,nscene) = diff_ang;
diff_mag_ar(:,nscene) = diff_mag;

MAPE_mag_ar(nscene,1) = MAPE_mag_sc;
MAE_ang_ar(nscene,1) = MAE_ang;

Result(:,2*nscene+1) = x_mag;
Result(:,2*nscene+2) = x_ang;

end
fprintf('MAPE is in percentage, and MAE is in degrees \n');
MAPE_mag = (Sum_mag_err_per_sc/(NoOfScene*NoOfState))*100
MAE_ang = Sum_ang_err_sc/(NoOfScene*NoOfState)
time = toc;
% save H_Iowa.mat H;

% Save MAPE & MAE for each bus to compare with DNN-SE results in Python
% writematrix(diff_mag_per*100, 'MAPE_per.csv');
% writematrix(diff_ang, 'MAE_per.csv');

