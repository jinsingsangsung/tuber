# main_all

This branch is to try different architectures to the video action detection task.
Total three models are try-able:
- TubeR: Tubelet Transformer for Video Action Detection
- SeqFormer
- Sparse R-CNN

Note that debugging is still ongoing.

This repo copied the supported code of [TubeR: Tubelet Transformer for Video Action Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_TubeR_Tubelet_Transformer_for_Video_Action_Detection_CVPR_2022_paper.pdf). 

```
# if running this code other than nsml, you can use docker image that contains pretrained models and files as well
docker run -it -v /data02/ava:/datasets  -v /home/jinsung/tuber:/tuber --gpus all --shm-size 32g --name tuber jinsingsangsung/tuber:1.2 /bin/bash 

# if running in nsml,
sh nsml_setup # brings pretrained models
sh nsml_setup_{gpu type} # brings AVA dataset to scratchpad

# example running command
python3 train_tuber_ava.py --config-file ./configuration/TubeR_CSN50_AVA21.yaml --num_gpu 4
python3 train_sparse_ava.py --config-file ./configuration/Sparse_CSN50_AVA21.yaml --num_gpu 4
python3 train_seqformer_ava.py --config-file ./configuration/SeqFormer_CSN50_AVA21.yaml --num_gpu 4
```
# Reproduction result of TubeR

Model | Dataset | Backbone | Backbone pretrained on | DETR pretrained on | #view | Original mAP | Reproduced mAP | config |
:-----: | :---: | :---: | :-----: | :-----: |  :---: | :----: | :---: | :---: |
| TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | 1 view | 27.2 |  **27.1** | [config](configuration/TubeR_CSN50_AVA21.yaml) |
| TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | 1 view | - | 24.98 | use focal loss |
| TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO* | 1 view | - | 0.0013 | original DETR weight |
| TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | - | 1 view | - | 25.03 | 30 epochs, lr: 2e-5|
| Deformable TubeR (vanilla) | AVA 2.1 | CSN-50 | Kinetics-400 | - | 1 view | - | TBD | [config](configuration/D2_TubeR_CSN50_AVA21.yaml) |


~~Still have no idea where the 1% drop comes from.~~ fixed with 8gpu (23.2.20)

# Model explanation
## Deformable TubeR (vanilla)
```
python3 train_d_tuber_ava2.py --config-file ./configuration/D2_TubeR_CSN50_AVA21.yaml
```
<img width="689" alt="Screen Shot 2023-03-03 at 2 01 53 PM" src="https://user-images.githubusercontent.com/76904126/222635641-4d7492c1-0b56-41a5-a59e-e19b99fb1bd1.png">

## Deformable TubeR: DETR + Deformable DETR
```
python3 train_d_tuber_ava3.py --config-file ./configuration/D3_TubeR_CSN50_AVA21.yaml
```
<img width="646" alt="Screen Shot 2023-03-03 at 4 01 03 PM" src="https://user-images.githubusercontent.com/76904126/222653064-10cf0cc4-33a4-4c7f-b358-a6d35527a229.png">

## Deformable TubeR: Deformable DETR + transformer
```
python3 train_d_tuber_ava4.py --config-file ./configuration/D4_TubeR_CSN50_AVA21.yaml
```
<img width="646" alt="Screen Shot 2023-03-03 at 11 52 36 PM" src="https://user-images.githubusercontent.com/76904126/222751737-ca8ce89a-88b7-4941-9aec-54af63e2822d.png">


## Citing TubeR
```
@inproceedings{zhao2022tuber,
  title={TubeR: Tubelet transformer for video action detection},
  author={Zhao, Jiaojiao and Zhang, Yanyi and Li, Xinyu and Chen, Hao and Shuai, Bing and Xu, Mingze and Liu, Chunhui and Kundu, Kaustav and Xiong, Yuanjun and Modolo, Davide and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13598--13607},
  year={2022}
}
```
