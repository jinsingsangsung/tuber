cd /mnt/ddn
cp -r frames.tar /mnt/tmp
cd /mnt/tmp
tar -xf frames.tar
rm -rf frames.tar
cd /home/nsml/tuber
python3 json_inspector.py
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-tuber-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-hier-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/og-dab-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/finetuned-og-dab-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/finetuned-dab-hier-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-sa-hier-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-sa-hoper-rmbin.pth ../pretrained_models/transformer/
