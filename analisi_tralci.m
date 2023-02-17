close all 
clear all
clc


figure_d = false;

currentFolder = pwd;
data_folder = "G:\Drive condivisi\AGRI-Grapevine-Pruning\progetto pruning volumi bernie\acquisizioni\20230124 ramo A variabili di inclinazione e distanza e background fixed\risultati\prova 1\cylindrificated"


% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(data_folder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', data_folder);
    uiwait(warndlg(errorMessage));
    data_folder = uigetdir(); % Ask for a new one.
    if data_folder == 0
         % User clicked Cancel
         return;
    end
end



% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(data_folder, '*.csv'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
T_total = []
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', baseFileName);
    % Now do whatever you want with this file name,
    % such as reading it in as an image array with imread()
    T = readtable(fullFileName,'ReadVariableNames',false, 'Delimiter', "'");
    T.Var1 = strrep( T.Var1,'[',' ');
    T.Var1 = strrep( T.Var1,']',' ');
    T.Var1 = strrep( T.Var1,'"','');
    %T.Var1 = strrep( T.Var1,"'","");
    T(1,:) = [];
    T.Var1 = split(T.Var1,',');
    T.Var1 = str2double(T.Var1);
    T = splitvars(T);
    T = renamevars(T,"Var1_1","frame");
    T = renamevars(T,"Var1_2","pixel");
    T = renamevars(T,"Var1_3","volume");
    T = renamevars(T,"Var1_4","distance");
    [number_of_row , number_of_column ] = size(T);
    column_with_cylinder = number_of_column - 4;
    iteration = column_with_cylinder/2;
    for it = 1:iteration
        T = renamevars(T,"Var1_" + int2str(it+4),"Vc_"+ int2str(it));
        T = renamevars(T,"Var1_" + int2str(it+4+iteration),"Dc_"+ int2str(it));



    end
    
    T_pulita = array2table(zeros(0,width(T)));
    T_pulita.Properties.VariableNames  = T.Properties.VariableNames;
   
    

    
    
    volumes = [];
    volumes_real = [];
    distances_meds = [];
    for j = 1:number_of_row
        %elimino out of range sensore saturo

        distance = T.distance(j);

        if distance > 280
            a = 1;
        elseif distance < 75
            a = 2;
        elseif j == 1
            a = 3;
        else
           T_pulita(end+1,:) = T(j,:);
        end 
    end
     [number_of_rowT , number_of_columnT ] = size(T_pulita);
    for j = 1:height(T_pulita)
        volume = 0;
        volume_real = 0;
        distances_med = []
        for w = 1:iteration
            volume_px_i = table2array(T_pulita(j,4+w));
            dist_i = table2array(T_pulita(j,4+iteration+w));
            volume_i = (volume_px_i); %qui la formula per calcolare il volume
            constant = 4.3841e-10;
            volume_i_real = (volume_px_i*(dist_i^3)) * constant; %qui la formula per calcolare il volume
            volume = volume + volume_i; %somma dei volumi iesimi
            volume_real = volume_real + volume_i_real; %somma dei volumi iesimi
            distances_med(end+1) = dist_i; % metto in un array distanze i esime
        end
        volumes(end+1,1) = volume;
        volumes_real(end+1,1) = volume_real;
        distances_meds(end+1,1) = mean(distances_med);
        
    
    end
    
    T_pulita(:,number_of_columnT +1) = table(distances_meds);
    T_pulita(:,number_of_columnT +2) = table(volumes);
    T_pulita(:,number_of_columnT +3) = table(volumes_real);
    T_pulita = renamevars(T_pulita,"Var37","distance_med");
    T_pulita = renamevars(T_pulita,"Var38","volume_tot");
    T_pulita = renamevars(T_pulita,"Var39","volume_tot_reale");
    T_total = [T_total;T_pulita];



    if figure_d
        figure(k)
        subplot(3,1,1)
        plot(volumes)
        title(baseFileName,"volume  [m3]")
        subplot(3,1,2)
        plot(T_pulita.volume)
        title("volume- from area to cylynder volume")
        subplot(3,1,3)
        plot(T_pulita.distance)
        title("distance")
        
        
        
    end




    
   
end

figure
subplot(2,1,1)
scatter(T_total.distance_med,T_total.volume_tot)
subplot(2,1,2)
scatter(T_total.distance_med,T_total.volume_tot_reale)


%%

close all 
clear all
clc

%modellazione errore
raggiF = [0.1 : 0.005 : 1.1];
Dmedi = [60: 1: 260];
Raggi_falsi = [];
Vol_reali = [];
Vol_fasulli = [];
errori_relativi = [];

for dis = 1:length(raggiF)
    

    L = 100;
    Rr = 1;

    DM = Dmedi(dis);
    e_r = (Rr/2)*(atan(Rr/DM)); %sperimentale
    ipo = DM - e_r + Rr;
    alpha = asin(Rr/ipo);
    Rf = Rr * cos(alpha);
    Vr = (Rr^2) * pi * L ;
    Vf = Vr * (cos(alpha))^2;

    Raggi_falsi(end+1) = Rf;
    Vol_reali(end+1) = Vr;
    Vol_fasulli(end+1) = Vf;
    errori_relativi(end+1) = e_r;

end

figure
subplot(3,1,1)
plot(Dmedi,Raggi_falsi)
hold on
yline(Rr)
legend('R false','R real')
title("raggio reale e fasullo")
xlabel("profondità cm")
ylabel("cm")
subplot(3,1,2)
plot(Dmedi,errori_relativi)
title("errore stima profondità")
xlabel("profondità cm")
ylabel("cm")
subplot(3,1,3)
plot(Dmedi,Vol_reali)
hold on
plot(Dmedi,Vol_fasulli)
title("volume reale e fasullo")
legend('Vreal','V fasullo')
xlabel("profondità cm")
ylabel("cm^3")

%voglio sapere come varia e_r  e la differenza tra raggi veri e finti e
%volumi veri e finti

%    alpha = asin(Rf / (DM + e_r)); %radiants
%     Rr = Rf / cos(alpha);
%     Rf = Rr * cos(alpha);
%     e_r = (Rr - (Rr*sin(alpha)))/2;
%     Vr = (Rr^2) * pi * L ;
%     Vr = (Rf^2)^2 / (cos(alpha))^2 * pi * L;
%     Vr = Vf / (cos(alpha))^2;

%%


close all 
clear all
clc


currentFolder = pwd;
%data_folder = "G:\Shared drives\AGRI-Grapevine\progetto pruning volumi\acquisizioni\20230120 provini cilindrici noti a diverse distanze\data_geometric20gen23\geometric";
data_folder = "G:\Drive condivisi\AGRI-Grapevine-Pruning\progetto pruning volumi bernie\acquisizioni\20230120 provini cilindrici noti a diverse distanze\data_geometric20gen23\geometric"
% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isfolder(data_folder)
    errorMessage = sprintf('Error: The following folder does not exist:\n%s\nPlease specify a new folder.', data_folder);
    uiwait(warndlg(errorMessage));
    data_folder = uigetdir(); % Ask for a new one.
    if data_folder == 0
         % User clicked Cancel
         return;
    end
end



% Get a list of all files in the folder with the desired file name pattern.
filePattern = fullfile(data_folder, '*.csv'); % Change to whatever pattern you need.
theFiles = dir(filePattern);
C_total = []
lengthT = 13.9;
diameter = 1.6;
area = lengthT* diameter;
volume  = ((diameter/2)^2)* pi * lengthT
for k = 1 : length(theFiles)
    baseFileName = theFiles(k).name;
    fullFileName = fullfile(theFiles(k).folder, baseFileName);
    fprintf(1, 'Now reading %s\n', baseFileName);
    C = readtable(fullFileName,'ReadVariableNames',false);
    [number_of_row , number_of_column ] = size(C);
    C_ord = [];
    for s = 1:number_of_row
        C_ord(s,1) = max([C.Var1(s),C.Var2(s)]);
        C_ord(s,2) = min([C.Var1(s),C.Var2(s)]);
        C_ord(s,3) = C.Var3(s);
        Area = C.Var1(s) * C.Var2(s);
        dimaeter = C_ord(s,2);
        altezza = C_ord(s,1);
        Volumec = ((dimaeter/2)^2)*pi * altezza;
        C_ord(s,4)= Area;
        C_ord(s,5) = Volumec;
    
    
    end
    C_total = [C_total;C_ord];


end

    dot = 10
    subplot(4,1,1)
    scatter(C_total(:,3), C_total(:,1),dot,'x')
    title("distanza vs lunghezza")
    subplot(4,1,2)

    scatter(C_total(:,3), C_total(:,2),dot,'x')
    title("distamnza vs diameter")
    subplot(4,1,3)
    scatter(C_total(:,3), C_total(:,4),dot,'x')
    title("distanza vs area")
    subplot(4,1,4)
    scatter(C_total(:,3), C_total(:,5),dot,'x')
    title("distanza vs volume")
    
    
    figure 
    subplot(3,1,1)
    plot((C_total(:,1) .* (C_total(:,3) )))
    title("length pixel mupltiply distance")
    
    subplot(3,1,2)
    plot((C_total(:,4) .* (C_total(:,3) .* C_total(:,3))))
    title("area pixel mupltiply distance ^ 2")
    
    
    subplot(3,1,3)
    plot((C_total(:,5) .* (C_total(:,3) .* C_total(:,3).* C_total(:,3))))
    title("volume pixel mupltiply distance ^ 3")
    

    

    mean_vp_x_D3 = mean((C_total(:,5) .* (C_total(:,3) .* C_total(:,3).* C_total(:,3))))
    constant = volume / mean_vp_x_D3
    volumeRR = (C_total(:,5).* (C_total(:,3) .* C_total(:,3).* C_total(:,3))) * constant;
    figure
    plot(volumeRR)
    title("Real volume from estimated constant")
    xlabel("prova n#")
    ylabel("volume [cm^3]")
    std = std(volumeRR)
    std_perc = std/volume * 100
    figure 
    normplot(volumeRR)
    figure 
    histogram(volumeRR - mean(volumeRR), 8)
    

%%

dati_geom_cilindro = "G:\Drive condivisi\AGRI-Grapevine\progetto pruning volumi\acquisizioni\20221213 volumetric calibration with depth\dATI GEOMETRICI CILINDRO.csv"
C = readtable(dati_geom_cilindro,'ReadVariableNames',false);

    figure(1)



 





%%
pause




T(1,:) = [];

% %%
% % optional cleaner
% 
% TT = T 
% leng = height(T)
% 
% 
% for i = 1 : leng
%     TT(end,:) = [];
% end
% 
% 
% %
% leng = height(T)
% 
% 
% for i = 1 : leng
% 
%     
%     a = T.distance_med;
%     
%     x = a(i,1);
%     if (x >= 110)
%         
%      
%     elseif (x < 90)
%         
%         
%     else
%         
%         TT(end + 1, :) = T(i,:)
%     end
% end
% 
% T = TT

%%


real_volume = [T.volume].*[(T.distance_med)];
vol_rel  = (T.volume - mean(T.volume))/ mean(T.volume);
dist_rel  = (T.distance_med - mean(T.distance_med))/ mean(T.distance_med);
vol_real_rel  = (real_volume - mean(real_volume))/ mean(real_volume);
dist_mean  = mean(T.distance_med);

real_volume_mean = mean(real_volume);
figure
subplot(3,1,1)
volume_mean = mean(T.volume)
plot(T.frame,T.volume)
title("volume")
subplot(3,1,2)

plot(T.frame,T.distance_med)
title("distance")
subplot(3,1,3)

plot(T.frame,real_volume)
title("real volume")



figure
k =M./([T.volume].* [T.distance_med]) ;


k_mean = mean(k);
relative_error = (k - k_mean)/k_mean;
kstd = std(k);
relative_std  = kstd / k_mean;
plot(T.frame,k)












