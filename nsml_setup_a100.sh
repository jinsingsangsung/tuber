cd /mnt/ddn
cp -r frames.tar /mnt/tmp
cd /mnt/tmp
tar -xf frames.tar
rm -rf frames.tar
cd /home/nsml/tuber
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-tuber-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-hier-detr.pth ../pretrained_models/transformer/