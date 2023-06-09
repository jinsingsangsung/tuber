cd ~/pretrained_models
mkdir main
cp -r ./transformer/dab-hoper-detr.pth ./main/dab_hoper.pth
cd main
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/ogv2_270-47_0507/checkpoints/ckpt_epoch_19.pth ./tuber.pth
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/Dab_hier_270-65/checkpoints/ckpt_epoch_16.pth ./baseline.pth
cd ~/tuber
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/ogv2_270-47_0507/res ./
mv ./res ./tuber_res
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/Dab_hier_270-65/res ./
mv ./res ./baseline_res
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/AVA_Tuber/Dab_hoper_270-88/res ./
mv ./res ./my_res