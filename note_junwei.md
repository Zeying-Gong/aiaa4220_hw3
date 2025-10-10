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

    # run the project 3 docker image
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
                # I trained for 16 hours on RTX 2060 laptop  and got 90% success date on training set

                # you can ignore all these errors:
                    malloc_consolidate(): invalid chunk size
                    Aborted (core dumped)

                # and if sometimes you see connection close errors, stop the container and start a new one will work

        # local validation and visualization using the minival split

            # You need to process the checkpoint to remove the aux loss model part

            # I select ckpt.15.pth for final testing according to validation performance

                (falcon) root@ai-precognition-laptop2:/app/Falcon# python process_ckp_for_eval.py evaluation/falcon/hm3d/checkpoints/ckpt.15.pth evaluation/falcon/hm3d/checkpoints/eval.pth

            # Now you can validate locally

                (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_mini.yaml habitat_baselines.num_environments=4 habitat.dataset.data_path=data/datasets/pointnav/social-hm3d/minival/minival.json.gz habitat_baselines.eval_ckpt_path_dir=evaluation/falcon/hm3d/checkpoints/eval.pth

            # Performance on Minival

                Average episode distance_to_goal: 1.7688 # lower better
                Average episode spl: 0.5812
                Average episode psc: 0.8965
                Average episode human_collision: 0.4000  # lower better
                Average episode success: 0.6000

            # compared to running eval with the pretrained pretrained_model
                (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_mini.yaml habitat_baselines.num_environments=4 habitat.dataset.data_path=data/datasets/pointnav/social-hm3d/minival/minival.json.gz habitat_baselines.eval_ckpt_path_dir=pretrained_model/pretrained_mini.pth

                Average episode distance_to_goal: 3.4038
                Average episode spl: 0.3735
                Average episode psc: 0.931
                Average episode human_collision: 0.5000
                Average episode success:0.4000

            # Visualize the validation episode with your policy

                # videos will be saved to evaluation/falcon/hm3d/video_checkpoint15
                    # each episode will be saved to one video, with the robot's observation and a top-down map shown (with the robot's location and the humans)


                (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_mini.yaml habitat_baselines.num_environments=4 habitat.dataset.data_path=data/datasets/pointnav/social-hm3d/minival/minival.json.gz habitat_baselines.eval_ckpt_path_dir=evaluation/falcon/hm3d/checkpoints/eval.pth habitat_baselines.video_dir=evaluation/falcon/hm3d/video_checkpoint15 habitat_baselines.eval.video_option=["disk"]


        # test and submit to eval.ai leaderboard

            # submission page: https://eval.ai/web/challenges/challenge-page/2650/submission

                # You need to download the mini submission template zip and put your checkpoint in it
                    # https://drive.google.com/file/d/1IRg5iPrWOOKKL6hTCWzZ2deQPLerBAMX/view

                    # decompress it, then replace the pretrained_model/pretrained_mini.pth with your processed .pth checkpoint file (and it needs to be the same file name, otherwise you should modify falcon_hm3d_minival_mini.yaml's eval_ckpt_path_dir)

                # Compress it
                    $ zip -r checkpoint.15.eval.zip checkpoint.15.eval.pth_run/

                # upload the zip file to the submission page, select Public, method name use Your Group Name+Your method name

            # check all your submission status: https://eval.ai/web/challenges/challenge-page/2650/my-submission

                # You can check the submission page's std output and std error file and check whether the run is successful

            # leaderboard page: https://eval.ai/web/challenges/challenge-page/2650/leaderboard
                # you can check off all other runs' "Show on Leaderboard" in "My Submission" page, so you can see the metrics on "Leaderboard" page

                 # minival should takes ~5minutes
                    # on eval.ai the minival results might be different due to huaman agent's random speed

                # We will mainly look at your results on the Test split
                    # submitting to Test split, the mini model should takes ~30 minutes to process
                    # the pretrained_mini.pth model gets 45% SR on the leaderboard
                    # my finetuned checkpoint.15.pth gets 47% SR on the leaderboard

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

        # train with 1 GPU (1080 TI) (will run validation along the way)
            # modify config/faster-rcnn_r50_fpn_giou_20e.py to train with batch_size=2;
            # takes up ~6GB GPU memory, 10 hours to train for 20 epochs

            (aiaa4220) junweil@ai-precognition-machine12:~/projects/aiaa4220/project2/resource/mm$ bash tools/dist_train.sh config/faster-rcnn_r50_fpn_giou_20e.py 1

            # this will download resnet-50 imagenet model as the feature encoder and train the whole faster rcnn model
            # too slow.  download the model first
                ~/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth

            # checkpoint saved to /home/junweil/projects/aiaa4220/project2/resource/mm/work_dirs/faster-rcnn_r50_fpn_giou_20e

            # I trained for 10 hours and got Epoch(val) [20][1000/1000]    coco/bbox_mAP: 0.7630

        # test and submit to Kaggle leaderboard
            (aiaa4220) junweil@ai-precognition-machine12:~/projects/aiaa4220/project2/resource/mm$ bash tools/dist_test.sh config/faster-rcnn_r50_fpn_giou_20e.py work_dirs/faster-rcnn_r50_fpn_giou_20e/epoch_20.pth 1

            # test result convert to kaggle submission format
                # paste the code from https://www.kaggle.com/competitions/hkustgz-aiaa-4220-2025-fall-project-2/overview into convert_to_submission.py
                    # or use this https://github.com/Zeying-Gong/aiaa4220_hw3/blob/main/convert_to_submission.py

                # Return to project root
                    (aiaa4220) junweil@ai-precognition-machine12:~/projects/aiaa4220/project2$ python aiaa4220_hw3/convert_to_submission.py --pred resource/mm/work_dirs/faster-rcnn_r50_fpn_giou_20e/test.bbox.json --test resource/data/test.json --output resource/submission.csv

            # submit submission.csv via the website
                # https://www.kaggle.com/competitions/hkustgz-aiaa-4220-2025-fall-project-2/submissions
                # Done! mAP 0.78
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

