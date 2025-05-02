

echo 'export PROJECT_ID=YOUR_PROJECT_ID' >> ~/.bashrc
echo 'export TPU_NAME=YOUR_TPU_VM_NAME' >> ~/.bashrc
echo 'export ZONE=YOUR_ZONE' >> ~/.bashrc
echo 'export RUNTIME_VERSION=YOUR_VM_RUNTIME(EX:tpu-ubuntu2204-base)' >> ~/.bashrc
echo 'export ACCELERATOR_TYPE=YOUR_VM_TYPE(EX:v4-32)' >> ~/.bashrc
source ~/.bashrc

ssh-keygen -t rsa -f ~/.ssh/id_rsa -N ""

# Add the public key to your TPU VM's metadata
cat ~/.ssh/id_rsa.pub
<HR>

wget https://raw.githubusercontent.com/ayaka14732/llama-2-jax/18e9625f7316271e4c0ad9dea233cfe23c400c9b/podrun
chmod +x podrun

nano ~/podips.txt (important: DO NOT COPY & PASTE, YOU NEED TO TYPE IN YOUR OWN TPU VM IP ADDRESSES )
```
172.21.12.86
172.21.12.87
172.21.12.83
```

pip3 install fabric

## TESTING
./podrun -iw -- echo meow
<HR>

./podrun -i -- sudo apt-get update -y -qq
./podrun -i -- sudo apt-get upgrade -y -qq
./podrun -- sudo apt-get install -y -qq nfs-common
sudo apt-get install -y -qq nfs-kernel-server
sudo mkdir -p /nfs_share
sudo chown -R nobody:nogroup /nfs_share
sudo chmod 777 /nfs_share


sudo nano /etc/exports (important: DO NOT COPY & PASTE, YOU NEED TO TYPE IN YOUR OWN MAIN TPU VM IP ADDRESS )
```
/nfs_share  10.130.0.0/24(rw,sync,no_subtree_check)
```

sudo exportfs -a
sudo systemctl restart nfs-kernel-server


./podrun -- sudo mkdir -p /nfs_share
./podrun -- sudo mount 10.130.0.9:/nfs_share /nfs_share
./podrun -i -- ln -sf /nfs_share ~/nfs_share


nano ~/nfs_share/setup.sh
```
rm -rf GPT-SoVITS-TPU
sudo apt-get install -y build-essential cmake python3.10-venv
sudo apt-get install -y ffmpeg cmake git-lfs zip unzip
git clone https://github.com/JungyongHan/GPT-SoVITS-TPU.git
cd GPT-SoVITS-TPU
mkdir GPT_weights
mkdir SoVITS_weights

sudo chsh $USER -s /usr/bin/bash
python -m venv ~/venv
. ~/venv/bin/activate
bash install.sh --source HF
mkdir logs
cp -r ~/nfs_share/logs/* logs/
```
chmod +x ~/nfs_share/setup.sh
./podrun -i ~/nfs_share/setup.sh


nano ~/nfs_share/kill_proc.sh
```
#!/bin/bash
kill_processes_using_device() {
    local device=$1
    pids=$(sudo lsof $device | grep python | awk '{print $2}' | sort -u)
    if [ -z "$pids" ]; then
        return 0
    fi
    for pid in $pids; do
        sudo kill -15 $pid
        sleep 0.5
        if ps -p $pid > /dev/null; then
            echo "[$device] PID $pid 정상 종료 실패, 강제 종료 시도..."
            sudo kill -9 $pid
            sleep 1
            
            if ps -p $pid > /dev/null; then
                echo "[$device] PID $pid 강제 종료 실패!"
            else
                echo "[$device] PID $pid 강제 종료 성공!"
            fi
        else
            echo "[$device] PID $pid 정상 종료 성공!"
        fi
    done
}

for i in {0..3}; do
    device="/dev/accel$i"
    kill_processes_using_device $device
done
```
chmod +x ~/nfs_share/kill_proc.sh
~/nfs_share/kill_proc.sh





:usefull example

#run multiple lines script to All VM example
nano ~/nfs_share/gitpull.sh
```
#!/bin/bash
cd GPT-SoVITS-TPU
git pull
```
chmod +x ~/nfs_share/gitpull.sh
./podrun -i ~/nfs_share/gitpull.sh

#check TPU VM online
./podrun -ic -- ~/venv/bin/python -c "import torch_xla.core.xla_model as xm; import torch_xla.runtime as xr; xr.global_ordinal() == 0 and print(xm.get_xla_supported_devices())"

#Move config to VM's local before starting script
./podrun -i cp ~/nfs_share/tmp_s2.json ~/
cd ~/GPT-SoVITS-TPU && ../podrun -icw -- ~/venv/bin/python ./GPT_SoVITS/s2_train.py --config ~/tmp_s2.json

#catch all core Rank Number(means Tensorcore num)
./podrun -ic -- ~/venv/bin/python -c "import os; os.environ['PJRT_DEVICE'] = 'TPU';import torch_xla.core.xla_model as xm;device = xm.xla_device();rank = xm.get_ordinal();print(f'My device: {device}, My rank: {rank}')"

#you can also use gloud command if you had set up env 
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all --command="git clone -b r2.5 https://github.com/pytorch/xla.git"
gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
  --zone=${ZONE} \
  --project=${PROJECT_ID} \
  --worker=all \
  --command="PJRT_DEVICE=TPU python3 ~/xla/test/test_train_mp_imagenet.py  \
  --fake_data \
  --model=resnet50  \
  --num_epochs=1 2>&1 | tee ~/logs.txt"
  
