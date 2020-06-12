# surrogate-pt-Bayeslands
Surrogate assisted parallel tempering Bayeslands



### code

We have two major versions 

* Surrogate parallel tempering using random-walk proposal distribution for Badlands landscape evolution model: [surrogate_ptBayeslands.py] to run with [run.sh] 
* paper online: [Surrogate-assisted Bayesian inversion for landscape and basin evolution models](https://arxiv.org/abs/1812.08655)



### Prerequisites

The framework is built using: 

* [Parallel tempering Bayeslands](https://github.com/intelligentEarth/pt-Bayeslands)
* [R. Chandra et. al, Multicore Parallel Tempering Bayeslands for Basin and Landscape Evolution](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GC008465) with  [Arxiv open access](https://arxiv.org/abs/1806.10939)
* [Badlands model](https://github.com/badlands-model/badlands) 
  

### Installation

* install conda python 3.x version: https://www.anaconda.com/distribution/

### installation -  linux terminal commands - in conda evironment

* conda install -c plotly chart-studio (install in conda env)
* curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"  (install pip3 in conda env)
* python3 get-pip.py --user
* pip3  install tribad
* pip3 install gFlex
* pip3 install badlands/  (do sudo if needed)

### run
* run_uplift.sh
* results in Examples/australia folder










 

###  Experiments 

* [Results with data for surrogate evaluation offline](https://github.com/intelligentEarth/surrogate-pt-Bayeslands/tree/master/SurrogateEvaluation)
* [Results for landscape evolution problems - selected runs](https://github.com/intelligentEarth/surrogate-pt-Bayeslands/tree/master/Examples)
 
 

## Versioning
 
* TBA

## Authors
 
* R. Chandra, D. Azam , A. Kapoor and R. D. Muller [Surrogate-assisted Bayesian inversion for landscape and basin evolution models](https://arxiv.org/abs/1812.08655)

## License

* This project is licensed under the MIT License - see the [Open Source Licence](https://opensource.org/licenses/MIT) file for details

## Acknowledgments

* Dr. Nathanial Butterworth, Sydney Informatics Hub, University of Sydney
* Mr. Konark Jain, Indian Institute of Technology

## Contact

* Dr. Rohitash Chandra, University of New South Wales (c.rohitash at gmail.com or rohitash.chandra at unsw.edu.au)
