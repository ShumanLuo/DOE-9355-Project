clc
clear all
close all
%% Three-Phase Linear State Estimate for Iowa system:
NodeData = xlsread('Iowa_Node_Data.xlsx'); % Have the node data sorted based upon the bus numbers 
LineData = xlsread('Iowa_Line_Data.xlsx','Line_Data_Final');

% load data_2500_fixed.mat;
% Vm_Ar  = str2double(node_voltage_names_separated);
% Vm_Ar = horzcat(Vm_Ar,voltage_error_full_ready);
% Vm_Ar = sortrows(Vm_Ar,1);
% Vm_Tr_Ar  = str2double(node_voltage_names_separated);
% Vm_Tr_Ar = horzcat(Vm_Tr_Ar,voltage_actual_full_ready);
% Vm_Tr_Ar = sortrows(Vm_Tr_Ar,1);
% Im_Ar = str2double(All_currents_WLS_names_ready);
% Im_Ar = horzcat(Im_Ar,All_currents_WLS_ready_to_send);

% load data_2500_fixed_true.mat;
% Vm_Ar  = str2double(node_voltage_names_separated);
% Vm_Ar = horzcat(Vm_Ar,voltage_actual_full_ready);
% Vm_Ar = sortrows(Vm_Ar,1);
% Vm_Tr_Ar  = str2double(node_voltage_names_separated);
% Vm_Tr_Ar = horzcat(Vm_Tr_Ar,voltage_actual_full_ready);
% Vm_Tr_Ar = sortrows(Vm_Tr_Ar,1);
% Im_Ar = str2double(All_currents_WLS_names_ready);
% Im_Ar = horzcat(Im_Ar,All_currents_WLS_actual_ready_to_send);

% Vm_Ar = xlsread('240_node_test_system_EXP_VOLTAGES.xlsx','Volt_final');
% Vm_Ar = sortrows(Vm_Ar,1);
% Im_Ar = xlsread('240_node_test_system_EXP_ElemCurrents.xlsx','Current_final');
% Vm_Tr_Ar = Vm_Ar;
% Im_Tr_Ar = Im_Ar;

load data_6000_GMM.mat;
Vm_Ar = VoltBusNameToBusNumber(node_voltage_names_separated);
Vm_Ar = horzcat(Vm_Ar,voltage_actual_full_ready);
% Vm_Ar = horzcat(Vm_Ar,voltage_error_full_ready);
Vm_Ar = sortrows(Vm_Ar,2);
Vm_Ar = sortrows(Vm_Ar,1);

Vm_Tr_Ar  = VoltBusNameToBusNumber(node_voltage_names_separated);
Vm_Tr_Ar = horzcat(Vm_Tr_Ar,voltage_actual_full_ready);
Vm_Tr_Ar = sortrows(Vm_Tr_Ar,2);
Vm_Tr_Ar = sortrows(Vm_Tr_Ar,1);

Im_Ar = CurrentBusNameToBusNumber(All_currents_WLS_names_ready);
Im_Ar = horzcat(Im_Ar,All_currents_WLS_actual_ready_to_send);
% Im_Ar = horzcat(Im_Ar,All_currents_WLS_ready_to_send);

nscene = 1;
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

tic;
BusMap = 1:length(NodeData);
NoOfNode = length(NodeData);
PerMileConstant = 5280;

[row, col] = size(LineData);
Sbase = 100*10^6;

Vbase1 = 13.8*10^3;
Ibase1 = Sbase/(sqrt(3)*Vbase1);
Zbase1 = (Vbase1^2)/Sbase;

for i = 1:row
    for k = 1:length(NodeData)
        if LineData(i,1)==NodeData(k)
            Fpos = k;
        end
        if LineData(i,2)==NodeData(k)
            Tpos = k;
        end
    end
    LineDataNew(i,1) = BusMap(Fpos);
    LineDataNew(i,2) = BusMap(Tpos);  
    
    if LineData(i,4)==100
        % This is a Switch
%         R = 0.0000001*eye(3,3);
%         X = zeros(3,3);
%         B = zeros(3,3);        
        alpha = cosd(120) + 1i*sind(120);
        A = [1, 1, 1; alpha^2, alpha, 1; alpha, alpha^2, 1]; 
        Zseq = 0.0001*eye(3); % Zseq(2,2) = 0;        
        Z =  A*Zseq*inv(A); 
        B = zeros(3,3);
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    
    if LineData(i,4)==101
        % This is "OH_3p_type1" line Config
        % The per mile values
        R =[0.615927	0.170927	0.170927;
            0.170927	0.615927	0.170927;
            0.170927	0.170927	0.615927];
        X = [1.209389	0.433188	0.433188;
            0.433188	1.209389	0.433188;
            0.433188	0.433188	1.209389];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = zeros(3,3)*10^(-9);                
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;               
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
   
    if LineData(i,4)==102
        % This is "OH_2p_type2" line Config
        % The per mile values
        R =[0.589255	0.169495;
            0.169495	0.589255];
        X = [1.074856	0.387876;
             0.387876	1.074856];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = zeros(2,2)*10^(-9);
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;         
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    
    if LineData(i,4)==103
        % This is OH_1p_type2 line Config
        % The per mile values
        R = [0.592473];            
        X = [1.065821];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = zeros(1,1)*10^(-9);
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;                
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    
    if LineData(i,4)==104
        % This is OH_1p_type5 line Config
        % The per mile values
         R =[1.19426];            
        X = [1.20542];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = zeros(1,1)*10^(-9);
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;        
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    
    if LineData(i,4)==105
        % This is OH_3p_type5 line Config
        % The per mile values
        R =[1.191602	0.234849	0.234849;
            0.234849	1.191602	0.234849;
            0.234849	0.234849	1.191602];            
        X = [1.209452	0.489263	0.489263;
             0.489263	1.209452	0.489263;
             0.489263	0.489263	1.209452];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = zeros(3,3)*10^(-9);
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;        
        Z = Z/Zbase1;
        B = B*Zbase1;
    end   
    
    if LineData(i,4)==106
        % This is UG_3p_type1 line Config
        % The per mile values
        R =[1.009423	0.409732	0.409732;
            0.409732	1.009423	0.409732;
            0.409732	0.409732	1.009423];            
        X = [0.493164	0.100849	0.100849;
            0.100849	0.493164	0.100849;
            0.100849	0.100849	0.493164];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = [286.101593	0	0;
                0	286.101593	0;
                0	0	286.101593]*10^(-9);
        B = 1i*2*pi*60*C;    
        B = B*LineData(i,3)/PerMileConstant;        
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    
    if LineData(i,4)==107
        % This is UG_3p_type2 line Config
        % The per mile values
        R =[1.692577	0.550087	0.550087;
            0.550087	1.692577	0.550087;
            0.550087	0.550087	1.692577];            
        X = [0.791756	0.344931	0.344931
             0.344931	0.791756	0.344931
             0.344931	0.344931	0.791756];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = [207.880539	0	0;
                0	207.880539	0;
                0	0	207.880539]*10^(-9);
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;        
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    
    if LineData(i,4)==108
        % This is UG_1p_type2 line Config
        % The per mile values
        R = [1.675338];            
        X = [1.210007];  
        Z = R + 1i*X; 
        Z = Z*LineData(i,3)/PerMileConstant;
        C = [207.880539]*10^(-9);
        B = 1i*2*pi*60*C;
        B = B*LineData(i,3)/PerMileConstant;        
        Z = Z/Zbase1;
        B = B*Zbase1;
    end
    [rw, cl] = size(Z);
    Z_ar([1:rw],[1:cl],i) = Z;
    B_ar([1:rw],[1:cl],i) = B;    
end

%% Find H_volt:
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

% "TotMeas" is the total number of voltage measurements:
% "sum(PhPerNode)" is the total number of power system states:
% Dimension of H_volt is "TotMeas"x"sum(PhPerNode)"
TotMeas = 0;
for i = 1:length(uPMUloc)
    TotMeas = TotMeas + PhPerNode(uPMUloc(i));    
end

H_V = zeros(TotMeas,sum(PhPerNode));
% H_V = zeros(1,sum(PhPerNode));

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
LineMonitor = xlsread('PMU_Line_Iowa.xlsx');
count = 1;
count_split = 1;
TotalPhase = sum(PhPerNode);
H_I = zeros(1,TotalPhase);

for i = 1:length(LineMonitor(:,1))
    Bus1 = LineMonitor(i,1);
    Bus2 = LineMonitor(i,2);
    
    % Search for this line in the LineDataNew array:
    for j = 1:length(LineDataNew(:,1))
        if ((Bus1==LineDataNew(j,1)) && (Bus2==LineDataNew(j,2))) || ((Bus1==LineDataNew(j,2)) && (Bus2==LineDataNew(j,1)))
            pos = j;
            NoOfPh = LineData(j,5);
            Z_sel = Z_ar([1:NoOfPh],[1:NoOfPh],j);
            B_sel = B_ar([1:NoOfPh],[1:NoOfPh],j);
            break;
        end        
    end
    Y_sel = inv(Z_sel);
    % First reduce the dimension of the Z matrix based upon number of
    % phases present in line (PhInLine):
    PhInLine = []; countPh = 1;
    for c = 2:4
        if (PhInNode(Bus1,c)~=0 && PhInNode(Bus2,c)~=0)
            PhInLine(countPh) = c-1;
            countPh = countPh+1;
        end            
    end        
    num_ph = length(Y_sel);
        
    % Find the total number of phases present till respective nodes:
    if (Bus1==1)
        Num_Ph_Bus1 = 0;
    else
        Num_Ph_Bus1 = sum(PhPerNode([1:Bus1-1]));
    end
    if (Bus2==1)
        Num_Ph_Bus2 = 0;
    else
        Num_Ph_Bus2 = sum(PhPerNode([1:Bus2-1]));
    end
    
    C1 = B_sel*(1/2)+Y_sel;
    C2 = (-1)*Y_sel;
    % Depending upon PhInLine determine column locations:
        % Find column locations for Bus 1:
        Col_Bus1 = [];nB1 = 1;
        for p = 1:length(PhInLine) 
            Col_Bus1(nB1) = PhInNode(Bus1,PhInLine(p)+1);
            nB1 = nB1+1;
        end
        % Find column locations for Bus 2:
        Col_Bus2 = [];nB2 = 1;
        for p = 1:length(PhInLine) 
            Col_Bus2(nB2) = PhInNode(Bus2,PhInLine(p)+1);
            nB2 = nB2+1;
        end
        
    H_I([count:count+num_ph-1],Col_Bus1) = C1;
    H_I([count:count+num_ph-1],Col_Bus2) = C2;           
    
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
H = vertcat(H_V,H_I);
% Below is for Least Squares (LS):
H_pseudo = ((H'*H)^-1)*(H');
x = H_pseudo*Meas;

% x = inv(H)*Meas;
% Below is for Weighted Least Squares (WLS):
% Sigma = 0.01;
% NoOfMeas = length(Meas);
% W = eye(NoOfMeas,NoOfMeas);
% W = (1/(Sigma^2))*W;
% H_pseudo = ((H'*inv(W)*H)^-1)*(H'*inv(W));
% x = H_pseudo*Meas;

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
diff_mag_per = diff_mag./xtr_mag;
sum_diff_mag_per = sum(diff_mag);
n = length(diff_mag_per);
MAPE_mag = (sum_diff_mag_per/n)*100

x_ang = angle(x)*180/pi;
xtr_ang = angle(xtr)*180/pi;
diff_ang = abs(xtr_ang-x_ang);
sum_diff_ang = sum(diff_ang);
n = length(diff_ang);
MAE_ang = sum_diff_ang/n

time = toc;

% save H_Iowa.mat H;
save H_Iowa_multiple_pmu_same_bus.mat H;
