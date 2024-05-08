%this script calcultes chance generated firing rate differences across
%trials. It uses the firing ana trials from decoder vars
clear all
close all
cd('C:\Users\Nancy\Documents\MATLAB\CSHL');

list=dir;




 %
for n=[12]; %folder number corresponds to animal folder

    cd(list(n).name); %goes to the first folder within the current folder
  (list(n).name)%prints name of current folder

    
    x=dir('*12-08_10sec');
 for nn=1:length(x);
  cd (x(nn).name)
  (x(nn).name)%priints 
 load 'decoder_vars.mat'
       
n=0:.001:3000;% time bins from the spike_train2 that has been loaded

permutations=30000;
laseron=laseron(laseron<3050 & laseron>10);
rmatrix=NaN(permutations,length(laseron)*2);
rdiff=NaN(size(spike_train,1),permutations);
FRoff=NaN(size(spike_train,1),length(laseron));
FRon=NaN(size(spike_train,1),length(laseron));
realFRdiff=NaN(size(spike_train,1),1);
realFRratio=NaN(size(spike_train,1),1);

bootp10=NaN(size(spike_train,1),1);
critval=NaN(size(spike_train,1),1);
halft=length(laseron);
for i=1:size(spike_train,1)
t = n(spike_train(i,:)==1);% s
for trialnum=1:length(laseron)
   
    zeroedraster=t(1:end)-laseron(trialnum);
     yy=zeroedraster(zeroedraster<0 & zeroedraster>-10);
    FRoff(i,trialnum)=length(yy)/10;
      yy=zeroedraster(zeroedraster<10 & zeroedraster>0);
    FRon(i,trialnum)=length(yy)/10;
  
end
realFRdiff(i,1)=mean(FRon(i,:))-mean(FRoff(i,:));
realFRratio(i,1)=mean(FRon(i,:))./mean(FRoff(i,:));
FR=[FRoff(i,:), FRon(i,:)];%firing rate per trial light off first then all light on
   for xx=1:30000
    ind=randperm(length(laseron)*2);
    rmatrix(xx,:)=FR(1,ind);
  
   end
   rdiff(i,:)=nanmean(rmatrix(:,1:halft),2)-nanmean(rmatrix(:,halft:end),2);
%    bootp10(i,1)=(min(sum(rdiff)<=realdiff(rep)),sum(diffdist{rep,type}>=realdiff(rep)))./nperms)*2;
   critval(i,1)=quantile(rdiff(i,:),0.975);
end
save 'randomFRchanges' critval rdiff realFRdiff realFRratio
 cd ..
 end
 cd ..
end

    