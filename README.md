# Unsupervised Classifier

## Local setup instructions

In `~/.ssh/config`

```
# NYU HPC config
# first we create the tunnel, with instructions to pass incoming
# packets on ports 8024, 8025 and 8026 through it and to specific
# locations
Host hpcgwtunnel
   HostName gw.hpc.nyu.edu
   ForwardX11 no
   LocalForward 8025 dumbo.hpc.nyu.edu:22
   LocalForward 8026 prince.hpc.nyu.edu:22
# next we create an alias for incoming packets on the port. The
# alias corresponds to where the tunnel forwards these packets
Host dumbo
  HostName localhost
  Port 8025
  ForwardX11 yes

Host prince
  HostName localhost
  Port 8026
  ForwardX11 yes
```

## To connect to HPC

### In a terminal window run:

`ssh YOURNETID@hpcgwtunnel`
then log in with your NYU password
this will connect you to the bastion which allows your computer to connect to the to the prince HPC server

### In a second terminal window run:

`ssh YOURNETID@prince`
then log in with your NYU password

## Set up the environment on Prince

### Create a new ssh key to connect to github

Run this command
`ssh-keygen -t rsa -b 4096 -C "goldmichael@gmail.com" -f ~/.ssh/github`

Print the generated key
`cat ~/.ssh/github.pub`

Select and copy that output from that command (the public key) to the clipboard

Add a new key to github here and paste in the public key here: https://github.com/settings/keys

In `~/.ssh/config` add
```
Host github.com
  IdentityFile ~/.ssh/github
```

This will use your new ssh key to connect to github.com

### Clone the github repo to your home folder on Prince

run this command
`git clone git@github.com:bmahlbrand/unsupervised-classifier.git`


### Set up the deep learning environment on Prince

First, change to the project directory
`cd unsupervised-classifier`

Then load the initialization script:
`./init.sh`

This will load the cuda and conda modules and set up the python environtment "deep" from environment.yml

## Start the deep learning environment on Prince

First, change to the project directory
`cd unsupervised-classifier`

Then load the initialization script:
`./start.sh`

This will load the cuda and conda modules, load the python environment "deep," and request gpus on slurm. 

From there you can execute python scripts
