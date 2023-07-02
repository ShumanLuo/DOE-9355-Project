function [ Current_name ] = CurrentBusNameToBusNumber( All_currents_WLS_names_ready )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
[row, ~] = size(All_currents_WLS_names_ready);
Current_name = [];

for r = 1:row    
   Bus_name_1 = All_currents_WLS_names_ready{r,1};
   Bus_name_2 = All_currents_WLS_names_ready{r,2};   
   Ph_val =  All_currents_WLS_names_ready{r,3};
   
   %% Check for Bus_name_1
   % Check if this is Bus 1
   flag_bus1 = 0;
   if length(Bus_name_1)==4 
        if (Bus_name_1=='bus1')
            flag_bus1 = 1;
        end
   end
   
   % Check if this is Transformer
   flag_transf = 0;
   if length(Bus_name_1)==4 
       if (Bus_name_1=='Xfmr')
            flag_transf = 1;
       end       
   end
   
   if (flag_bus1==1)
       bus_num = 1;
       Current_name(r,1) = bus_num;       
       
   elseif (flag_transf==1)
       bus_num = 9999;
       Current_name(r,1) = bus_num;
              
   else
       Bus_name_sel = Bus_name_1(4:7);
       bus_num = str2double(Bus_name_sel);
       Current_name(r,1) = bus_num;
              
   end
   
   %% Check for Bus_name_2
   % Check if this is Bus 1
   flag_bus1 = 0;
   if length(Bus_name_2)==4 
        if (Bus_name_2=='bus1')
            flag_bus1 = 1;
        end
   end
   
   % Check if this is Transformer
   flag_transf = 0;
   if length(Bus_name_2)==4 
       if (Bus_name_2=='Xfmr')
            flag_transf = 1;
       end       
   end
   
   if (flag_bus1==1)
       bus_num = 1;
       Current_name(r,2) = bus_num;       
       
   elseif (flag_transf==1)
       bus_num = 9999;
       Current_name(r,2) = bus_num;
              
   else
       Bus_name_sel = Bus_name_2(4:7);
       bus_num = str2double(Bus_name_sel);
       Current_name(r,2) = bus_num;
              
   end   
   Current_name(r,3) = str2double(Ph_val);   
   
end


end

