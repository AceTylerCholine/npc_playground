%this script calcultes chance generated firing rate differences across
%trials with a sliding window of 5 ms from 0-150ms after laser onset. It
%uses the trials from decoder vars. Baseline firing is calculated for every
%trial as window of 100 ms before light onset. 
clear all
close all
cd('C:\Users\Nancy\Documents\MATLAB\CSHL');

list=dir;




 %
for j=[13 14  17 18 19 20]; %folder number corresponds to animal folder

    cd(list(j).name); %goes to the first folder within the current folder
  (list(j).name)%prints name of current folder

    
    x=dir('*5mspulses');
    
    bin=.005;
    edges=0:bin:.150;
 for nn=1:length(x);
  cd (x(nn).name)
  (x(nn).name)%priints 
 load 'decoder_vars.mat'
       
n=0:.001:2000;%time bins from the spike_train2 that has been loaded

permutations=1000;
laseron=laseron(laseron>1);
rmatrix=NaN(permutations,length(laseron)*2);
rdiff=NaN(size(spike_train,1),permutations);
FRoff=NaN(size(spike_train,1),length(laseron));
FRon=NaN(size(spike_train,1),length(laseron));
realFRdiff=NaN(size(spike_train,1),length(edges));
realFRratio=NaN(size(spike_train,1),length(edges));
  perc=NaN(size(spike_train,1),length(edges));
 pvalb=NaN(size(spike_train,1),length(edges));
 
 inhibited=zeros(size(spike_train,1),1);
 excited=zeros(size(spike_train,1),1);
 
 
bootp10=NaN(size(spike_train,1),1);
critval=NaN(size(spike_train,1),length(edges));
halft=length(laseron);
for i=1:size(spike_train,1)
    
t = n(spike_train(i,1:2000001)==1);% s
if length(t)<80
    
else
for e=1:length(edges)-1
for trialnum=1:length(laseron)
   
    zeroedraster=t(1:end)-laseron(trialnum);
     yy=zeroedraster(zeroedraster<0 & zeroedraster>-.100);%%
    FRoff(i,trialnum)=length(yy)/bin;
      yy=zeroedraster(zeroedraster<(edges(e+1)) & zeroedraster>(edges(e)));
    FRon(i,trialnum)=length(yy)/bin;
  
end
realFRdiff(i,e)=mean(FRon(i,:))-mean(FRoff(i,:));
realFRratio(i,e)=mean(FRon(i,:))./mean(FRoff(i,:));
FR=[FRoff(i,:), FRon(i,:)];%firing rate per trial light off first then all light on
   for xx=1:permutations;
    ind=randperm(length(laseron)*2);
    rmatrix(xx,:)=FR(1,ind);
  
   end
   rdiff(i,:)=nanmean(rmatrix(:,1:halft),2)-nanmean(rmatrix(:,halft:end),2);
%    bootp10(i,1)=(min(sum(rdiff)<=realdiff(rep)),sum(diffdist{rep,type}>=realdiff(rep)))./nperms)*2;
   critval(i,e)=quantile(rdiff(i,:),0.975);
   
   
  

ord=sort(rdiff(i,:));


[nn, ind]=histc(realFRdiff(i,e),ord);
if ind==0% this could happen if the real value is outside the range of values in bootstrap distribution
     if realFRdiff(i,e)>max(ord)
      perc(i,e)=100;
  
    elseif realFRdiff(i,e)<max(ord)
    perc(i,e)=0;
    end
else
perc(i,e)=(ind/permutations)*100;
end
pvalb(i,e)=sum(abs(rdiff(i,:))>=abs(realFRdiff(i,e)))./permutations;
end
      if sum(perc(i,:)<2.5)>1 & sum(pvalb(i,:)<0.05)>1
  inhibited(i,1)=1;     
      end

     if sum(perc(i,:)>97.5)>1 & sum(pvalb(i,:)<0.05)>1
  excited(i)=1;      
     end
end
end   

save 'randomFRchanges0-150msbaseline100ms' critval rdiff realFRdiff realFRratio perc pvalb edges inhibited excited
 cd ..
 end
 cd ..
end