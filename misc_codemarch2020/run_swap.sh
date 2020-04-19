
#!/bin/bash  
echo Running all 	 

problem=2
replica=4
swapint=10 # same as surogate interval 
#samples=4000
maxtemp=2
burn=0.25
pt_stage=0.5
raintimeint=4
initialtopoep=0.5
surrogate_probability=0.5

echo $problem 


 
for samples in  2000
	do  
 
			python srBayeslands_revamp_swap.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint -surrogate $surrogate_probability
			python realtime_visualise_results.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $initialtopoep -rain_intervals $raintimeint

  
  
	done 
 

