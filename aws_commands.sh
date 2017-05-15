#!/bin/bash
# Change the setting of your .pem file to read-only
chmod 400 <your-key-pair>.pem
ssh -i <your-key-pair>.pem ec2-user@<IP-addresss> # Login into the EC2 instance
./Miniconda3-latest-Linux-x86_64.sh # Install the Miniconda version
# Append this to .bash_profile
# added by Miniconda3 4.3.11 installer
export PATH="/home/ec2-user/miniconda3/bin:$PATH" 
# Sometimes you need to reboot the machine
sudo su
reboot
# After it is started, try if the installation has successed
conda -h
# Install the virtual environment to your EC2 instance
conda create -n <VirtualEnvName> python=3.5 # 3.5 is essential to work for tensorflow
# Use WinScp to copy your data to the machine or you could also use git
# For using github
sudo yum install git
git clone https://github.com/<git-user>/<git-repo>
# Copying your code to EC2 instance using WinSCP for Windows (download at https://winscp.net/eng/download.php) for Mac, instructions [TBA]
# On the WinSCP login, under Host-Name copy the IP address from EC2 console
# Port number 22 remains unchanged
# User-name is always ec2-user
# Go to Advanced setting on WinSCP, and under SSH--Authentication, select the private-key file that you got it from AWS, this will
# convert the .pem to Putty compatiable format called .ppk. Then click okay and you should be able to use WinSCP

# Tunnel the notebook to your local machine
ssh -i <your-key-pair>.pem -L 8080:localhost:8888 ec2-user@<IP address>

# Activate the virtual environment
source activate <VirtualEnvName>
# Then type jupyter notebook
jupyter notebook
# Copy the token that is displayed on terminal
# On the local machine
http://localhost:8080
# paste the token in the password brower

echo "Great ! you are all set to use AWS EC2 instance"


