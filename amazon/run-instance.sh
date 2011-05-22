#! /bin/bash
#Fire up an ubuntu small machine in the western US. We have an EBS in the 1a availability zone
if [ -z "`ec2-describe-instances|grep running`" ]; then
    echo "Starting EC2 instance ..."
    ec2-run-instances ami-19bfef5c --instance-type m1.small --region us-west-1 -z us-west-1a -k edward-ec2-hashmapd
    while [ -z "`ec2-describe-instances|grep running`" ]; do
        echo "Waiting for instance to be ready ..."
    done
fi

#Wait a bit and then associate the ip address and volume
EC2_INSTANCE=`ec2-describe-instances|grep running|awk '{print $2}'|head -1`
EC2_VOLUME=`ec2-describe-volumes|awk '{print $2}'|head -1`
EC2_ADDR=`ec2-describe-addresses|awk '{print $2}'|head -1`
EC2_SSH="ssh ubuntu@${EC2_ADDR} -A -i ${EC2_DEFAULT_PEM} -o StrictHostKeyChecking=no"
ec2-associate-address ${EC2_ADDR} -i ${EC2_INSTANCE}
ec2-attach-volume ${EC2_VOLUME} -i ${EC2_INSTANCE} -d /dev/sdh
echo "Instance ${EC2_INSTANCE} available at ${EC2_ADDR}"


#If there is a conflict with the known hosts, then delete the offending line
#TODO: this dosn't seem to be working ...
KNOWN_HOSTS=${EC2_SSH} "" |& grep Offending | awk '{print $4}'
if [ -n "${KNOWN_HOSTS}" ]; then
    echo "Remove conflicting key,  ${KNOWN_HOSTS}"
    KNOWN_HOSTS_FILE=`python -c "print '${KNOWN_HOSTS}'.split(':')[0]"`
    KNOWN_HOSTS_LINE=`python -c "print '${KNOWN_HOSTS}'.split(':')[1]"`
    vim -es +${KNOWN_HOSTS_LINE} +d +wq ${KNOWN_HOSTS_FILE}
fi

echo "Mount EBS storage ${EC2_VOLUME} on /ebs"
#Set up block storage
${EC2_SSH} "sudo mkdir /ebs"
${EC2_SSH} "sudo mount -t ext3 /dev/sdh /ebs"
${EC2_SSH} "sudo chown ubuntu:ubuntu /ebs"

echo "Install software"
#Install software
${EC2_SSH} "sudo aptitude update -y"
${EC2_SSH} "sudo aptitude install git-core python-setuptools python-dev gcc -y"

echo "Setup couchbase server"
${EC2_SSH} "wget http://c3145442.r42.cf0.rackcdn.com/couchbase-server-community_x86_1.1.deb"
${EC2_SSH} "sudo dpkg -i couchbase-server-community_x86_1.1.deb"
${EC2_SSH} "sudo mkdir /mnt/couchdb"
${EC2_SSH} "sudo chown couchbase:couchbase /mnt/couchdb /ebs/couchdb /ebs/log/couchdb"
${EC2_SSH} "sudo echo -e '[couchdb]\ndatabase_dir = /ebs/couchdb/\nview_index_dir = /mnt/couchdb/\n\n[log]\nfile = /ebs/log/couchdb/couch.log\nlevel = error\n' > /opt/couchbase-server/etc/couchdb/local.ini"

echo "Install hashmapd software"
${EC2_SSH} "sudo easy_install pip"
${EC2_SSH} "sudo pip install couchapp couchdb tweepy config"
${EC2_SSH} "git clone git@github.com:utunga/hashmapd.git"

#Set up an ssh tunnel, so that you can access couch
echo "Setup ssh tunnel ... couch available on http://localhost:5994"
ssh -f -N -L 5994:localhost:5984 ubuntu@${EC2_ADDR} -i ${EC2_DEFAULT_PEM}

echo "To connect to the instance type:"
echo ${EC2_SSH}

