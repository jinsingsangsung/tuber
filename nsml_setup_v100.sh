cd /mnt/video-nfs5/datasets/ava
cp -r frames.tar /mnt/tmp
cd ../
cp -r JHMDB.tar /mnt/tmp
cp -r UCF101_v2.tar.gz.tar /mnt/tmp
cd /mnt/tmp
tar -xf frames.tar
rm -rf frames.tar
tar -xf JHMDB.tar
rm -rf JHMDB.tar
tar -xf UCF101_v2.tar.gz.tar
rm -rf UCF101_v2.tar.gz.tar
cd /home/nsml/tuber
python3 json_inspector.py
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-hier-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/og-dab-detr.pth ../pretrained_models/transformer/
# cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/finetuned-og-dab-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/finetuned-dab-hier-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-sa-hier-detr.pth ../pretrained_models/transformer/
cp -r /mnt/video-nfs5/users/jinsung/results/tubelet-transformer/dab-hoper-detr.pth ../pretrained_models/transformer/