# Test note by Junwei Liang

```

    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ git clone https://github.com/Zeying-Gong/aiaa4220_hw3

    # data
    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tar -zxvf aiaa4220_hw3_data.tgz

    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ mv data/ aiaa4220_hw3/Falcon/

    # pretrained mini model is already in Falcon/pretrained_model/pretrained_mini.pth

    # check if any docker container is already running. It is good to remove containers if not using them
        $ sudo docker ps -ls

    # run the project 3 image
        $ sudo docker run --name homework -it --gpus all --network host      --runtime=nvidia      --entrypoint /bin/bash      -w /app/Falcon      -v /home/junweil/projects/aiaa4220/aiaa4220_hw3/Falcon/:/app/Falcon zeyinggong/robosense_socialnav:v0.7


        # After entering docker container

            root@ai-precognition-laptop2:/app/Falcon# source activate falcon


        (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_train_mini_junwei.yaml


    # 用tensorboard查看进度
        $ pip install tensorboard
        (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tensorboard --logdir=aiaa4220_hw3/Falcon/training/falcon/hm3d/tb --bind_all

        # 打开浏览器 http://localhost:6006 看训练集的success rate; 在tb里有smoothing的选项，可以看看平滑后的曲线整体趋势，是否上升，选取拐点附近的checkpoint，比如这里选择1.5M step的

        # checkpoint

```
