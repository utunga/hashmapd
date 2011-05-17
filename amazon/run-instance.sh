#! /bin/bash
#Fire up an ubuntu small machine in the western US. We have an EBS in the 1a availability zone
ec2-run-instances ami-19bfef5c --instance-type m1.small --region us-west-1 -z us-west-1a -K $EC2_DEFAULT_PEM -k $EC2_PRIVATE_KEY

#Wait a bit and then associate the ip address and volume
EC2_VOLUME=`ec2-describe-volumes|awk '{print $2}'|head -1`
EC2_INSTANCE=`ec2-describe-instances|grep running|awk '{print $2}'|head -1`
EC2_ADDR=`ec2-describe-addresses|awk '{print $2}'|head -1`
ec2-associate-address $EC2_ADDR -i $EC2_INSTANCE
ec2-attach-volume $EC2_VOLUME -i $EC2_INSTANCE -d /dev/sdh

#Lets head over to our new machine
ssh ubuntu@$EC2_ADDR -i $EC2_DEFAULT_PEM
#The first time, you need to format the ebs ... #sudo mfs.ext3 /dev/sdh
#Now mount the ebs storage
sudo mkdir /ebs
sudo mount -t ext3 /dev/sdh /ebs
sudo chown /ebs ubuntu

