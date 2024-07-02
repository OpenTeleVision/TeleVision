<h1 align="center"><img src="img/logo.png" width="40"  style="vertical-align: middle;"> Open-TeleVision: Teleoperation with
Immersive Active Visual Feedback</h1>
Xuxin Cheng*, Jialong Li*, Shiqi Yang, Ge Yang, 
Xiaolong Wang

UC San Diego, MIT

[Video](https://x.com/xuxin_cheng/status/1783838823849546067)

<p align="center">
<img src="./img/television.jpg" width="80%"/>
</p>

## Introduction
Stream head, hand, wrist data from VisionPro or Meta Quest 3. Stream real-time stereo video from camera to VR Devices. 

## Installation

```bash
conda create -n tv python=3.8
conda activate tv
pip install -r requirements.txt
cd act/detr && pip install -e .
```

If you want to try teleoperation example with an active cam with zed camera (teleop_active_cam.py):

Install zed sdk: https://www.stereolabs.com/developers/release/

If you want to try teleoperation example in a simulated environment (teleop_hand.py):

Install Isaac Gym: https://developer.nvidia.com/isaac-gym/


## Teleoperation Guide

### Local streaming
Apple does not allow WebXR on non-https connections. To test the application locally, we need to create a self-signed certificate and install it on the client. You need a ubuntu machine and a router. Connect the VisionPro and the ubuntu machine to the same router. 
1. install mkcert: https://github.com/FiloSottile/mkcert
2. check local ip address: 

```
ifconfig | grep inet
```
Suppose the local ip address of the ubuntu machine is `192.168.8.102`.

3. create certificate: 

```
mkcert -install && mkcert -cert-file cert.pem -key-file key.pem 192.168.8.102 localhost 127.0.0.1
```

4. open firewall on server
```
sudo iptables -A INPUT -p tcp --dport 8012 -j ACCEPT
sudo iptables-save
sudo iptables -L
```
or can be done with `ufw`:
```
sudo ufw allow 8012
```
5.
```python
self.app = Vuer(host='0.0.0.0', cert="./cert.pem", key="./key.pem")
```

6. install ca-certificates on VisionPro
```
mkcert -CAROOT
```
Copy the rootCA.pem via AirDrop to VisionPro and install it.

Settings > General > About > Certificate Trust Settings. Under "Enable full trust for root certificates", turn on trust for the certificate.

settings > Apps > Safari > Advanced > Feature Flags > Enable WebXR Related Features

7. open the browser on Safari on VisionPro and go to `https://192.168.8.102:8012?ws=wss://192.168.8.102:8012`

8. Click `Enter VR` and ``Allow`` to start the VR session.

### Network Streaming
For Meta Quest3, installation of the certificate is not trivial. We need to use a network streaming solution. We use `ngrok` to create a secure tunnel to the server. This method will work for both VisionPro and Meta Quest3.

1. Install ngrok: https://ngrok.com/download
2. Run ngrok
```
ngrok http 8012
```
3. Copy the https address and open the browser on Meta Quest3 and go to the address.

## Training Guide
1. Download dataset from <link>.

2. Place the downloaded dataset in ``data/recordings/``.

3. Process the specified dataset for training using ``scripts/post_process.py``.

4. You can verify the image and action sequences of a specific episode in the dataset using ``scripts/replay_demo.py``.

5. To train ACT, run:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt
```

6. After finished training, save jit for the desired checkpoint:
```
    python imitate_episodes.py --policy_class ACT --kl_weight 10 --chunk_size 60 --hidden_dim 512 --batch_size 45 --dim_feedforward 3200 --num_epochs 50000 --lr 5e-5 --seed 0 --taskid 00 --exptid 01-sample-expt\
                               --save_jit --resume_ckpt 25000
```

7. You can visualize the trained policy with input from dataset using ``scripts/deploy_sim.py``.

## Citation
```
@article{cheng2024tv,
title={Open-TeleVision: Teleoperation with Immersive Active Visual Feedback},
author={Cheng, Xuxin and Li, Jialong and Yang, Shiqi and Yang, Ge and Wang, Xiaolong},
journal={arXiv preprint arXiv:2407.01512},
year={2024}
}
```