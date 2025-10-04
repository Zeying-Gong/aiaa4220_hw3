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

            # num_env=2, step 1.5M, 20 checkpoints
            (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_train_mini_junwei.yaml
                # you can ignore all the  SSD Load Failure!
                # takes 20 minutes before training starts


    # Use tensorboard to check your training progress. Check for losses and the success rate
        $ pip install tensorboard
        (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tensorboard --logdir=aiaa4220_hw3/Falcon/training/falcon/hm3d/tb --bind_all

        # open a browser http://localhost:6006
            # 在tb里有smoothing的选项，可以看看平滑后的曲线整体趋势，是否上升，选取拐点附近的checkpoint，比如这里选择1.5M step的

        # checkpoint

```

+ Running P2
```
    # get data and code from https://www.kaggle.com/competitions/hkustgz-aiaa-4220-2025-fall-project-2/data
        # 4GB data
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

