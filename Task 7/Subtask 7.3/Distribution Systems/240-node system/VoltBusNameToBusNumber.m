function [ Volt_name  ] = VoltBusNameToBusNumber( node_voltage_names_separated )

[row, ~] = size(node_voltage_names_separated);
Volt_name = [];
for r = 1:row
   Bus_name = node_voltage_names_separated{r,1};
   Ph_val =  node_voltage_names_separated{r,2};
   
   % Check if this is Bus 1
   flag_bus1 = 0;
   if length(Bus_name)==4 
        if (Bus_name=='bus1')
            flag_bus1 = 1;
        end
   end
   
   % Check if this is Transformer
   flag_transf = 0;
   if length(Bus_name)==8 
       if (Bus_name=='bus_xfmr')
            flag_transf = 1;
       end       
   end
   
   if (flag_bus1==1)
       bus_num = 1;
       Volt_name(r,1) = bus_num;
       Volt_name(r,2) = str2double(Ph_val);
       
   elseif (flag_transf==1)
       bus_num = 9999;
       Volt_name(r,1) = bus_num;
       Volt_name(r,2) = str2double(Ph_val);
       
   else
       Bus_name_sel = Bus_name(4:7);
       bus_num = str2double(Bus_name_sel);
       Volt_name(r,1) = bus_num;
       Volt_name(r,2) = str2double(Ph_val);
   end
         
end

end

