close all 
clear all
clc


currentFolder = pwd
data_folder = "G:\Drive condivisi\AGRI-Grapevine\progetto pruning volumi\acquisizioni\20221213 volumetric calibration with depth\volume_and_distances"
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
    distances = []
    for j = 1:number_of_row
        distance = mean(table2array(T(j,21:36)));
        distances(end+1,1) = distance;
        
       




    end

    figure(k)
    subplot(3,1,1)
    plot(T.pixel)
    title(baseFileName,"pixel")
    subplot(3,1,2)
    plot(T.volume)
    title("volume- from area to cylynder volume")
    subplot(3,1,3)
    plot(T.distance)
    title("distance")




    
   
end

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












