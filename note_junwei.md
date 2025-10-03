# Test note by Junwei Liang

```


    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ git clone https://github.com/Zeying-Gong/aiaa4220_hw3 --recursive

    # data
    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tar -zxvf aiaa4220_hw3_data.tgz

    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ mv data/ aiaa4220_hw3/Falcon/

    # pretrained model
    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220/aiaa4220_hw3$ mkdir Falcon/pretrained_model/
    (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220/aiaa4220_hw3$ cp pretrained_mini.pth Falcon/pretrained_model/

    $ sudo docker run --name homework -it --gpus all --network host      --runtime=nvidia      --entrypoint /bin/bash      -w /app/Falcon      -v /home/junweil/projects/aiaa4220/aiaa4220_hw3/Falcon/:/app/Falcon zeyinggong/robosense_socialnav:v0.7


    # 进入docker container后

    root@ai-precognition-laptop2:/app/Falcon# source activate falcon

    # 2个env 占用3GB显存, 4个env占用5.5 GB显存; GPU利用率~10%，CPU 20%,内存占用14GB
        # 4个env好像跑到死机掉驱动了
        # 还是用2个env吧

        (falcon) root@ai-precognition-laptop2:/app/Falcon# python -u -m habitat-baselines.habitat_baselines.run --config-name=social_nav_v2/falcon_hm3d_train_mini.yaml habitat_baselines.num_environments=2

        # 修改训练的config, 用junwei


    # 用tensorboard查看进度
        $ pip install tensorboard
        (base) junweil@ai-precognition-laptop2:~/projects/aiaa4220$ tensorboard --logdir=aiaa4220_hw3/Falcon/training/falcon/hm3d/tb --bind_all

        # 打开浏览器 http://localhost:6006 看训练集的success rate; 在tb里有smoothing的选项，可以看看平滑后的曲线整体趋势，是否上升，选取拐点附近的checkpoint，比如这里选择1.5M step的

        # checkpoint存在


    # 本地从mini model finetune, 然后测试
        # 1. 获取代码
        # 2.

```
