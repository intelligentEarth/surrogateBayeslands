
#!/bin/bash  
echo Running all 	 
 
replica=8
swapint=3 
samples=1000
maxtemp=2
burn=0.25
pt_stage=0.5
raintimeint=4 

echo $problem 

  
for problem in  1  #1 2
 	do
	for surrogate_interval in  0.10 #0.05 0.10 0.15  # this is for surrogate interval 
  		do
		for surrogate_probability in  0.4 #0.2 0.4 0.6 0.8 
			do   
 
			python srBayeslands_revamp_.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $surrogate_interval -rain_intervals $raintimeint -surrogate $surrogate_probability
			python realtime_visualise_results.py -p $problem -s $samples -r $replica -t $maxtemp -swap $swapint -b $burn -pt $pt_stage  -epsilon $surrogate_interval -rain_intervals $raintimeint
    
  
	done  
	done
done


 