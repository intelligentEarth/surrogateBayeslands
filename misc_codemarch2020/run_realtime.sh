
#!/bin/bash  
echo Running all 	 

problem=2
replica=8
swapint=40 # same as surogate interval 
#samples=4000
maxtemp=2
burn=0.25
pt_stage=0.5
raintimeint=4
initialtopoep=0.5
surrogate_probability=0 

echo $problem 


 
for samples in    5000 5000 5000 5000 5000
	do  
 
			python srBayeslands_revamp_.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint -surrogate $surrogate_probability
			python realtime_visualise_results.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint

  
  
	done 

problem=2
replica=8
swapint=40 # same as surogate interval 
#samples=4000
maxtemp=2
burn=0.25
pt_stage=0.5
raintimeint=4
initialtopoep=0.5
surrogate_probability=0.5

echo $problem 


 
for samples in  5000 5000 5000 5000 5000
	do  
 
			python srBayeslands_revamp_.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint -surrogate $surrogate_probability
			python realtime_visualise_results.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint

  
  
	done 



