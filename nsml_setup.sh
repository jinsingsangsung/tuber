cd /home/nsml
cp -r /mnt/video-nfs5/users/jinsung/tubelet-transformer ./ # bring the code
mkdir pretrained_models && cp -r /mnt/video-nfs5/users/jinsung/pretrained_models/tubelet-transformer ./pretrained_models # bring the pretrained_model
cp -r /mnt/video-nfs5/datasets/hmdb51.tar /mnt/tmp # copy the dataset
cd /mnt/tmp && tar -xvf JHMDB.tar && rm -rf JHMDB.tar # untar the dataset

# if needed
# cd /home/nsml/pretrained_models/transformer
# wget https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/detr.pth
# cd /home/nsml/pretrained_models/backbone
# wget https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/irCSN_50_ft_kinetics_from_ig65m_f233743920.mat
# cd /home/nsml/pretrained_models/main
# wget https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/TubeR_CSN152_JHMDB.pth # case of JMDB

cd /home/nsml/tubelet-transformer/datasets
wget https://yzaws-data-log.s3.amazonaws.com/shared/TubeR_cvpr22/assets.zip
unzip assets.zip && rm -rf assets.zip