#! /bin/bash
#Fire up an ubuntu small machine in the western US. We have an EBS in the 1a availability zone
#ec2-run-instances ami-19bfef5c --instance-type m1.small --region us-west-1 -z us-west-1a -k edward-ec2-hashmapd

#Wait a bit and then associate the ip address and volume
EC2_VOLUME=`ec2-describe-volumes|awk '{print $2}'|head -1`
EC2_INSTANCE=`ec2-describe-instances|grep running|awk '{print $2}'|head -1`
EC2_ADDR=`ec2-describe-addresses|awk '{print $2}'|head -1`
EC2_SSH="ssh ubuntu@${EC2_ADDR} -i ${EC2_DEFAULT_PEM}"
ec2-associate-address ${EC2_ADDR} -i ${EC2_INSTANCE}
ec2-attach-volume ${EC2_VOLUME} -i ${EC2_INSTANCE} -d /dev/sdh

#Set up block storage
${EC2_SSH} "sudo mkdir /ebs"
${EC2_SSH} "sudo mount -t ext3 /dev/sdh /ebs"
${EC2_SSH} "sudo chown ubuntu:ubuntu /ebs"

#Install software
${EC2_SSH} "sudo aptitude update -y"
${EC2_SSH} "sudo aptitude install couchdb git-core python-setuptools python-dev gcc -y"
${EC2_SSH} "sudo easy_install pip"
${EC2_SSH} "sudo pip install couchapp couchdb tweepy config"
${EC2_SSH} -A "git clone git@github.com:utunga/hashmapd.git"

#Set the couch config so that the database files go to /ebs and the index files go to /mnt
scp -i ${EC2_DEFAULT_PEM} local.ini  ubuntu@${EC2_ADDR}:~
${EC2_SSH} "sudo mv ~/local.ini /etc/couchdb/local.ini"
${EC2_SSH} "sudo chown couchdb:couchdb /etc/couchdb/local.ini"
${EC2_SSH} "sudo mkdir -p /ebs/var/lib/couchdb/0.10.0 /mnt/var/lib/couchdb/0.10.0 /ebs/var/log/couchdb/"
${EC2_SSH} "sudo chown couchdb:couchdb /ebs/var/lib/couchdb/0.10.0 /mnt/var/lib/couchdb/0.10.0 /ebs/var/log/couchdb/"
${EC2_SSH} "sudo chmod a+r /ebs/var/lib/couchdb/0.10.0 /mnt/var/lib/couchdb/0.10.0 /ebs/var/log/couchdb/"
${EC2_SSH} "sudo /etc/init.d/couchdb stop"
${EC2_SSH} "sudo /etc/init.d/couchdb start"

#Now load the views
${EC2_SSH} "cd ~/hashmapd/couchdb; python create_db.py"

#Set up an ssh tunnel, so that you can access couch
ssh -f -N -L 5994:localhost:5984 ubuntu@${EC2_ADDR} -i ${EC2_DEFAULT_PEM}


