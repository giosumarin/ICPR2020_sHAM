########## PRUNING ##########
python pruning.py 0 30
python pruning.py 0 40
python pruning.py 0 50
python pruning.py 0 60
python pruning.py 0 70
python pruning.py 0 80
python pruning.py 0 90
python pruning.py 0 95
python pruning.py 0 96
python pruning.py 0 97
python pruning.py 0 98
python pruning.py 0 99

########## WEIGHTSHARING ##########
python weightsharing.py 0 0.001 2 2 2
python weightsharing.py 0 0.001 2 32 2
python weightsharing.py 0 0.001 2 128 2
python weightsharing.py 0 0.001 2 1024 2
python weightsharing.py 0 0.001 32 2 2
python weightsharing.py 0 0.001 32 32 2
python weightsharing.py 0 0.001 32 128 2
python weightsharing.py 0 0.001 32 1024 2
python weightsharing.py 0 0.001 128 2 2
python weightsharing.py 0 0.001 128 32 2
python weightsharing.py 0 0.001 128 128 2
python weightsharing.py 0 0.001 128 1024 2
python weightsharing.py 0 0.001 1024 2 2
python weightsharing.py 0 0.001 1024 32 2
python weightsharing.py 0 0.001 1024 128 2
python weightsharing.py 0 0.001 1024 1024 2

python weightsharing.py 0 0.001 2 2 32
python weightsharing.py 0 0.001 2 32 32
python weightsharing.py 0 0.001 2 128 32
python weightsharing.py 0 0.001 2 1024 32
python weightsharing.py 0 0.001 32 2 32
python weightsharing.py 0 0.001 32 32 32
python weightsharing.py 0 0.001 32 128 32
python weightsharing.py 0 0.001 32 1024 32
python weightsharing.py 0 0.001 128 2 32
python weightsharing.py 0 0.001 128 32 32
python weightsharing.py 0 0.001 128 128 32
python weightsharing.py 0 0.001 128 1024 32
python weightsharing.py 0 0.001 1024 2 32
python weightsharing.py 0 0.001 1024 32 32
python weightsharing.py 0 0.001 1024 128 32
python weightsharing.py 0 0.001 1024 1024 32

########## PRUNING WEIGHTSHARING ##########
####### FIXED PRUNING
python pruning_weightsharing.py 0 0.001 60 2 2 2
python pruning_weightsharing.py 0 0.001 60 2 32 2
python pruning_weightsharing.py 0 0.001 60 2 128 2
python pruning_weightsharing.py 0 0.001 60 2 1024 2
python pruning_weightsharing.py 0 0.001 60 32 2 2
python pruning_weightsharing.py 0 0.001 60 32 32 2
python pruning_weightsharing.py 0 0.001 60 32 128 2
python pruning_weightsharing.py 0 0.001 60 32 1024 2
python pruning_weightsharing.py 0 0.001 60 128 2 2
python pruning_weightsharing.py 0 0.001 60 128 32 2
python pruning_weightsharing.py 0 0.001 60 128 128 2
python pruning_weightsharing.py 0 0.001 60 128 1024 2
python pruning_weightsharing.py 0 0.001 60 1024 2 2
python pruning_weightsharing.py 0 0.001 60 1024 32 2
python pruning_weightsharing.py 0 0.001 60 1024 128 2
python pruning_weightsharing.py 0 0.001 60 1024 1024 2

python pruning_weightsharing.py 0 0.001 60 2 2 32
python pruning_weightsharing.py 0 0.001 60 2 32 32
python pruning_weightsharing.py 0 0.001 60 2 128 32
python pruning_weightsharing.py 0 0.001 60 2 1024 32
python pruning_weightsharing.py 0 0.001 60 32 2 32
python pruning_weightsharing.py 0 0.001 60 32 32 32
python pruning_weightsharing.py 0 0.001 60 32 128 32
python pruning_weightsharing.py 0 0.001 60 32 1024 32
python pruning_weightsharing.py 0 0.001 60 128 2 32
python pruning_weightsharing.py 0 0.001 60 128 32 32
python pruning_weightsharing.py 0 0.001 60 128 128 32
python pruning_weightsharing.py 0 0.001 60 128 1024 32
python pruning_weightsharing.py 0 0.001 60 1024 2 32
python pruning_weightsharing.py 0 0.001 60 1024 32 32
python pruning_weightsharing.py 0 0.001 60 1024 128 32
python pruning_weightsharing.py 0 0.001 60 1024 1024 32

####### FIXED WEIGHTSHARING
python pruning_weightsharing.py 0 0.001 30 32 32 2
python pruning_weightsharing.py 0 0.001 40 32 32 2
python pruning_weightsharing.py 0 0.001 50 32 32 2
python pruning_weightsharing.py 0 0.001 70 32 32 2
python pruning_weightsharing.py 0 0.001 80 32 32 2
python pruning_weightsharing.py 0 0.001 90 32 32 2
python pruning_weightsharing.py 0 0.001 95 32 32 2
python pruning_weightsharing.py 0 0.001 96 32 32 2
python pruning_weightsharing.py 0 0.001 97 32 32 2
python pruning_weightsharing.py 0 0.001 98 32 32 2
python pruning_weightsharing.py 0 0.001 99 32 32 2

########## PROBABILISTIC QUANTIZATION ##########
python stochastic.py 0 0.001 2 2 2
python stochastic.py 0 0.001 2 32 2
python stochastic.py 0 0.001 2 128 2
python stochastic.py 0 0.001 2 1024 2
python stochastic.py 0 0.001 32 2 2
python stochastic.py 0 0.001 32 32 2
python stochastic.py 0 0.001 32 128 2
python stochastic.py 0 0.001 32 1024 2
python stochastic.py 0 0.001 128 2 2
python stochastic.py 0 0.001 128 32 2
python stochastic.py 0 0.001 128 128 2
python stochastic.py 0 0.001 128 1024 2
python stochastic.py 0 0.001 1024 2 2
python stochastic.py 0 0.001 1024 32 2
python stochastic.py 0 0.001 1024 128 2
python stochastic.py 0 0.001 1024 1024 2

python stochastic.py 0 0.001 2 2 32
python stochastic.py 0 0.001 2 32 32
python stochastic.py 0 0.001 2 128 32
python stochastic.py 0 0.001 2 1024 32
python stochastic.py 0 0.001 32 2 32
python stochastic.py 0 0.001 32 32 32
python stochastic.py 0 0.001 32 128 32
python stochastic.py 0 0.001 32 1024 32
python stochastic.py 0 0.001 128 2 32
python stochastic.py 0 0.001 128 32 32
python stochastic.py 0 0.001 128 128 32
python stochastic.py 0 0.001 128 1024 32
python stochastic.py 0 0.001 1024 2 32
python stochastic.py 0 0.001 1024 32 32
python stochastic.py 0 0.001 1024 128 32
python stochastic.py 0 0.001 1024 1024 32

########## PRUNING PROBABILISTIC QUANTIZATION ##########
####### FIXED PRUNING
python pruning_stochastic.py 0 0.001 60 2 2 2
python pruning_stochastic.py 0 0.001 60 2 32 2
python pruning_stochastic.py 0 0.001 60 2 128 2
python pruning_stochastic.py 0 0.001 60 2 1024 2
python pruning_stochastic.py 0 0.001 60 32 2 2
python pruning_stochastic.py 0 0.001 60 32 32 2
python pruning_stochastic.py 0 0.001 60 32 128 2
python pruning_stochastic.py 0 0.001 60 32 1024 2
python pruning_stochastic.py 0 0.001 60 128 2 2
python pruning_stochastic.py 0 0.001 60 128 32 2
python pruning_stochastic.py 0 0.001 60 128 128 2
python pruning_stochastic.py 0 0.001 60 128 1024 2
python pruning_stochastic.py 0 0.001 60 1024 2 2
python pruning_stochastic.py 0 0.001 60 1024 32 2
python pruning_stochastic.py 0 0.001 60 1024 128 2
python pruning_stochastic.py 0 0.001 60 1024 1024 2

python pruning_stochastic.py 0 0.001 60 2 2 32
python pruning_stochastic.py 0 0.001 60 2 32 32
python pruning_stochastic.py 0 0.001 60 2 128 32
python pruning_stochastic.py 0 0.001 60 2 1024 32
python pruning_stochastic.py 0 0.001 60 32 2 32
python pruning_stochastic.py 0 0.001 60 32 32 32
python pruning_stochastic.py 0 0.001 60 32 128 32
python pruning_stochastic.py 0 0.001 60 32 1024 32
python pruning_stochastic.py 0 0.001 60 128 2 32
python pruning_stochastic.py 0 0.001 60 128 32 32
python pruning_stochastic.py 0 0.001 60 128 128 32
python pruning_stochastic.py 0 0.001 60 128 1024 32
python pruning_stochastic.py 0 0.001 60 1024 2 32
python pruning_stochastic.py 0 0.001 60 1024 32 32
python pruning_stochastic.py 0 0.001 60 1024 128 32
python pruning_stochastic.py 0 0.001 60 1024 1024 32

####### FIXED PROBABILISTIC QUANTIZATION
python pruning_stochastic.py 0 0.001 30 32 2 32
python pruning_stochastic.py 0 0.001 40 32 2 32
python pruning_stochastic.py 0 0.001 50 32 2 32
python pruning_stochastic.py 0 0.001 70 32 2 32
python pruning_stochastic.py 0 0.001 80 32 2 32
python pruning_stochastic.py 0 0.001 90 32 2 32
python pruning_stochastic.py 0 0.001 95 32 2 32
python pruning_stochastic.py 0 0.001 96 32 2 32
python pruning_stochastic.py 0 0.001 97 32 2 32
python pruning_stochastic.py 0 0.001 98 32 2 32
python pruning_stochastic.py 0 0.001 99 32 2 32
