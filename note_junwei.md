# Test note by Junwei Liang

+ Running P3
```

    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ git clone https://github.com/Zeying-Gong/aiaa4220_hw3

    # data
    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tar -zxvf aiaa4220_hw3_data_v2.tgz

    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ mv data/ aiaa4220_hw3/Falcon/

    # pretrained mini model is already in Falcon/pretrained_model/pretrained_mini.pth

    # check if any docker container is already running. It is good to remove containers if not using them
        $ sudo docker ps -ls

    # run the project 3 image
        $ sudo docker run --name homework -it --gpus all --network host      --runtime=nvidia      --entrypoint /bin/bash      -w /app/Falcon      -v /home/junweil/projects/aiaa4220/aiaa4220_hw3/Falcon/:/app/Falcon zeyinggong/robosense_socialnav:v0.7

        # After entering docker container

            root@ai-precognition-laptop2:/app/Falcon# source activate falcon

            (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_train_mini_junwei.yaml
                # Using the reduce training set with 15 scenes.
                # num_env=4, train for 80K steps, save 20 checkpoints, takes 5.3 GB GPU memory/8GB RAM,  CPU Util at ~20%, GPU Util at ~10%; takes ~20 hours


        # Use tensorboard to check your training progress. Check for losses and the success rate
            $ pip install tensorboard
            (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tensorboard --logdir=aiaa4220_hw3/Falcon/evaluation/falcon/hm3d/tb/ --bind_all

            # open a browser http://localhost:6006
                # 在tb里有smoothing的选项，可以看看平滑后的曲线整体趋势，是否上升，选取拐点附近的checkpoint

            # checkpoint saved at Falcon/evaluation/falcon/hm3d/checkpoints/

        # local validation and visualization

        # test and submit to eval.ai leaderboard

```

+ Running P2
```
    # get data and code from https://www.kaggle.com/competitions/hkustgz-aiaa-4220-2025-fall-project-2/data
        # 4GB data

        (base) junweil@ai-precognition-machine12:~/projects/aiaa4220/project2$ unzip ~/Downloads/hkustgz-aiaa-4220-2025-fall-project-2.zip

    # install env
        # I'm using 1080 TI, NVIDIA-SMI Driver Version: 575.64.03      CUDA Version: 12.9
        $ conda create -n aiaa4220 python=3.10
        $ pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
        $ pip install numpy==1.24
        $ pip install openmim
        $ mim install mmengine==0.10.7 mmcv==2.1.0 mmdet==3.3.0
        $ pip install pycocotools

        # train with 1 GPU (1080 TI)
            # modify config/faster-rcnn_r50_fpn_giou_20e.py to train with batch_size=2;
            # takes up ~6GB GPU memory, 10 hours to train for 20 epochs

            (aiaa4220) junweil@ai-precognition-machine12:~/projects/aiaa4220/project2/resource/mm$ bash tools/dist_train.sh config/faster-rcnn_r50_fpn_giou_20e.py 1

            # this will download resnet-50 imagenet model as the feature encoder and train the whole faster rcnn model
            # too slow.  download the model first
                ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

            # checkpoint saved to /home/junweil/projects/aiaa4220/project2/resource/mm/work_dirs/faster-rcnn_r50_fpn_giou_20e



        # val

        # test and submit to Kaggle leaderboard
```

+ Setting Up http server for data downloading
```
    $ wget -c https://precognition.team/shares/software/xampp-linux-x64-5.5.34-0-installer.run

    + Install. $ chmod +x ** sudo ./xampp-linux-x64-5.5.34-0-installer.run

    + /opt/lampp$ sudo ./xampp security

    + Set htdocs to other locations, add a shared folder other than the htdocs


    $ vi /opt/lampp/etc/httpd.conf

        #DocumentRoot "/opt/lampp/htdocs"
        DocumentRoot "/home/junweil/htdocs"
        <Directory "/home/junweil/htdocs">
        ...
        remove FollowSymLinks  Indexes

        Alias /shares "/home/junweil/htdocs_shares"
        <Directory "/home/junweil/htdocs_shares">
                Options Indexes
                Order allow,deny
                IndexOptions NameWidth=40 Charset=UTF-8
                Allow from all
                Require all granted

        </Directory>

    + Start: /opt/lampp$ sudo ./xampp restart

    # Now you can simply (on campus network)
        $ wget -c http://office.precognition.team/shares/aiaa4220_hw3_data_v2.tgz
        # when no one else is downloading it, 30GB should takes 5 minutes to download with 100MB/s
```

