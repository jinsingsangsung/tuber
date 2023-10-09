cd /mnt/video_nfs4/datasets/
cp -r JHMDB.tar /mnt/tmp
cp -r UCF101_v2.tar.gz.tar /mnt/tmp
cd /mnt/tmp
tar -xf JHMDB.tar
rm -rf JHMDB.tar
tar -xf UCF101_v2.tar.gz.tar
rm -rf UCF101_v2.tar.gz.tar
cd /home/nsml/tuber
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/dab-tuber-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/dab-hier-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/dab-hier-detr-jhmdb.pth ../pretrained_models/transformer/
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/og-dab-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/finetuned-og-dab-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/finetuned-dab-hier-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video_nfs4/users/jinsung/results/tubelet-transformer/dab-sa-hier-detr.pth ../pretrained_models/transformer/