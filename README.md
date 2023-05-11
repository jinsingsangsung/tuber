# main_all

This branch is to try different architectures to the video action detection task.

<details>
<summary> Previous models' brief summaries </summary>  
Total three models are try-able:
  
  - TubeR: Tubelet Transformer for Video Action Detection
  - SeqFormer
  - Sparse R-CNN
  ```
  # example running command
  python3 train_tuber_ava.py --config-file ./configuration/TubeR_CSN50_AVA21.yaml --num_gpu 4
  python3 train_sparse_ava.py --config-file ./configuration/Sparse_CSN50_AVA21.yaml --num_gpu 4
  python3 train_seqformer_ava.py --config-file ./configuration/SeqFormer_CSN50_AVA21.yaml --num_gpu 4
  ```
Note that debugging is still ongoing.

This repo copied the supported code of [TubeR: Tubelet Transformer for Video Action Detection](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhao_TubeR_Tubelet_Transformer_for_Video_Action_Detection_CVPR_2022_paper.pdf). 
</details>
  
```
# if running this code other than nsml, you can use docker image that contains pretrained models and files as well
docker run -it -v /data02/ava:/datasets  -v /home/jinsung/tuber:/tuber --gpus all --shm-size 32g --name tuber jinsingsangsung/tuber:1.2 /bin/bash 

# if running in nsml,
sh nsml_setup # brings pretrained models
sh nsml_setup_{gpu type} # brings AVA dataset to scratchpad

# now run the running command following the descriptions in the following sections.
```
# Experiments on TubeR variants
There have been many expriments before this, you can check these below:
<details>
  <summary> unsuccessful attempts </summary>
    
  ### mostly based on TubeR original architecture
    
  Model | Dataset | Backbone | Backbone pretrained on | DETR pretrained on | Original mAP | Reproduced mAP | config |
  :-----: | :---: | :---: | :-----: | :-----: | :----: | :---: | :---: |
  | TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | 27.2 |  **27.1** | [config](configuration/TubeR_CSN50_AVA21.yaml) |
  | TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | - | 24.98 | use focal loss |
  | TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO* | - | 0.0013 | original DETR weight |
  | TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 25.03 | 30 epochs, lr: 2e-5|
  | Deformable TubeR (vanilla) | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 20.39 | [config](configuration/D2_TubeR_CSN50_AVA21.yaml) |
  | DETR + DeformableDETR | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 21.99 | [config](configuration/D3_TubeR_CSN50_AVA21.yaml) |
  | DeformableDETR + transformer| AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 20.48 | [config](configuration/D4_TubeR_CSN50_AVA21.yaml) |
  | DN-DAB-Deformable TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 18.13 | [config](configuration/Dab_DN_D_TubeR_CSN50_AVA21.yaml) |
  | DETR + DN-DAB-Deformable transformer | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | TBD | [config](configuration/Dab2_DN_D_TubeR_CSN50_AVA21.yaml) |
  | TubeR + CSCE | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | - | 26.82 | [config](configuration/new_TubeR_CSN50_AVA21.yaml) |
  | TubeR + CSCE | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 22.78 | - |
  | TubeR + FCWG | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 20.56 | - |
  | TubeR + ZT | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | TBD | - |
  | Cloca TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | *COCO*, AVA | - | 21.57 | [config](configuration/cloca_TubeR_CSN50_AVA21.yaml) |
  | Cloca TubeR | AVA 2.1 | CSN-50 | Kinetics-400 | - | - | 17.24 | - |
  
  ~~Still have no idea where the 1% drop comes from.~~ fixed with 8gpu (23.2.20)    
</details>

Currently doing experiments on AVA22, and trying on the new baseline

Model | Dataset | Backbone | Backbone pretrained on | DETR pretrained on | f-mAP | V-mAP | config |
:-----: | :---: | :---: | :-----: | :-----: | :----: | :---: | :---: |
| TubeR | AVA 2.2 | CSN-50 | Kinetics-400 | *COCO*, AVA | 29.2 | - | - |
| TubeR w/o LTC | AVA 2.2 | CSN-50 | Kinetics-400 | *COCO*, AVA | 27.7 <br> 27.6(rep) | - | - |
| Baseline | AVA 2.2 | CSN-50 | Kinetics-400 | *COCO*, AVA | 23.37 | - | - |
| Baseline + CE | AVA 2.2 | CSN-50 | Kinetics-400 | *COCO*, AVA | 27.43 | - | [config](configuration/Dab_hier_CSN50_AVA21.yaml) |
| Baseline + CE <br> (IC ver.)| AVA 2.2 | CSN-50 | Kinetics-400 | *COCO*, AVA | 27.11 | - | - |
| TubeR | JHMDB | CSN-152 | Kinetics-400 | *COCO*, AVA | N/A | 82.3 | - |
| Baseline + CE | JHMDB | CSN-152 | Kinetics-400 | *COCO*, AVA | 77.27 | 84.59 | [config](configuration/Dab_hier_CSN152_JHMDB.yaml)  |

# Model explanation

  ### New Baseline and Classification Embedding(CE) (5/5)
  1. Baseline
  2. Baseline + Classification Embedding

  #### Baseline
  Baseline now gets rid of TubeR ornaments, and use more basic architecture. 
  It uses alternative spatio-temporal encoding, and use basic Dab-DETR structure in the decoder; both loc and cls come out from the same feature.
  ![Screen Shot 2023-05-11 at 3 03 12 PM](https://media.oss.navercorp.com/user/36297/files/3aca1284-41c5-4a71-a038-f64b3166f1b1)
  Baseline code is not available here currently, but will be uploaded soon.
  
  #### Baseline + Classification Embedding (CE)
  CE is a embedding that holds the information for classification. It computes the offsets from the decoder cls embedding output, and re-compute the correlation between the global context and queries based on the offset-shifted positional prior.
  ![Screen Shot 2023-05-11 at 3 07 25 PM](https://media.oss.navercorp.com/user/36297/files/997f2975-f1dc-45bd-b5db-c798d1452757)

  ```
  python3 train_dab_hier.py
  ```


  ### Deformable TubeR family
  1. Deformable TubeR (vanilla)
  2. Deformable TubeR: DETR + Deformable DETR
  3. Deformable TubeR: Deformable DETR + transformer
  4. DN-DAB-Deformable TubeR
  5. DN-DAB-Deformable TubeR: DETR + DN-DAB-Deformable TubeR
<details>
  <summary> Deformable TubeR family model descriptions & How to run </summary>

  #### Deformable TubeR (vanilla)
  ```
  python3 train_d_tuber_ava2.py --config-file ./configuration/D2_TubeR_CSN50_AVA21.yaml
  ```
  <img width="646" alt="Screen Shot 2023-03-03 at 2 01 53 PM" src="https://user-images.githubusercontent.com/76904126/222635641-4d7492c1-0b56-41a5-a59e-e19b99fb1bd1.png">

  #### Deformable TubeR: DETR + Deformable DETR
  ```
  python3 train_d_tuber_ava3.py --config-file ./configuration/D3_TubeR_CSN50_AVA21.yaml
  ```
  <img width="646" alt="Screen Shot 2023-03-03 at 4 01 03 PM" src="https://user-images.githubusercontent.com/76904126/222653064-10cf0cc4-33a4-4c7f-b358-a6d35527a229.png">

  from March 9th, the model is modified to:

  <img width="646" alt="Screen Shot 2023-03-14 at 5 01 35 PM" src="https://user-images.githubusercontent.com/76904126/224934235-07318eb1-83ef-4861-942b-983f4c15c27c.png">

  Note that the reported accuracy is from the previous version.

  #### Deformable TubeR: Deformable DETR + transformer
  ```
  python3 train_d_tuber_ava4.py --config-file ./configuration/D4_TubeR_CSN50_AVA21.yaml
  ```
  <img width="646" alt="Screen Shot 2023-03-03 at 11 52 36 PM" src="https://user-images.githubusercontent.com/76904126/222751737-ca8ce89a-88b7-4941-9aec-54af63e2822d.png">

  #### DN-DAB-Deformable TubeR
  ```
  python3 train_dab_dn_d_tuber.py --config-file ./configuration/Dab_DN_D_TubeR_CSN50_AVA21.yaml
  ```
  <img width="646" alt="Screen Shot 2023-03-14 at 5 05 19 PM" src="https://user-images.githubusercontent.com/76904126/224935205-1e3aea32-2265-4c2a-9e6b-2ddaf7e71f0f.png">

  #### DN-DAB-Deformable TubeR: DETR + DN-DAB-Deformable TubeR
  ```
  python3 train_dab2_dn_d_tuber.py --config-file ./configuration/Dab2_DN_D_TubeR_CSN50_AVA21.yaml
  ```
  <img width="646" alt="Screen Shot 2023-03-14 at 5 10 29 PM" src="https://user-images.githubusercontent.com/76904126/224936539-d7ed48dc-f9a9-459e-aa9d-4e97097007c4.png">

</details>

  ### Classification + Localization (Cloca) family
  1. Cloca TubeR
  2. Cloca TubeR v2
  
<details>
  <summary> Cloca family model descriptions & How to run </summary>
    
  #### Cloca TubeR

  ```
  python3 train_cloca_tuber.py
  ```

  <img width="1251" alt="Screen Shot 2023-04-03 at 8 36 18 PM" src="https://user-images.githubusercontent.com/76904126/229498183-903b9a56-e7e7-492f-af3d-29cac36e51c3.png">

  #### Cloca TubeR v2
  But this model has some issues with its implementation: structure needs to be fixed
  ```
  python3 train_cloca_tuber2.py
  ```

  <img width="1251" alt="Screen Shot 2023-04-03 at 8 37 36 PM" src="https://user-images.githubusercontent.com/76904126/229498434-0cea4958-e38c-4946-a060-7dcbfbe9ea2a.png">
    
</details>


  ### Something else...
  1. new TubeR
  2. DAB-TubeR
  
<details>
  <summary> Descriptions & How to run </summary>
    
  #### New TubeR

  ```
  python3 train_new_tuber.py
  ```
  New TubeR contains three features: CSCE (class specific channel enhancement) module, FC weight regularization, and zoom transformer.
  - CSCE: each class has its own semi-independent network, following the structure of dynamic convolution. This is mainly inspired from SparseRCNN.
  - FC weight regularization: this makes the FC laer of the model more diverse across channels
  - zoom transformer: as the decoder layer proceeds: this enables the model to see the highly activated region once again. The layer number of the decoder is set to 2 due to the memory/time issues
  
  (3/31) currently the FC weight regularization seems to degrade the model accuracy, only CSCE / zoom transformer module is regarded.
  
  (4/1) current code only contains zoom transformer module: I am on the test to see if it is effective.

  #### DAB TubeR
  ```
  python3 train_dab_tuber.py
  python3 train_dab_gt_tuber.py
  ```
  This is an implementataion of DAB-DETR on the TubeR. The one with GT takes GT boxes as input when training, and does not take the GT boxes when testing.
  Also planning on the model that takes GT boxes on inference time as well.

  <img width="1251" alt="Screen Shot 2023-04-03 at 8 37 36 PM" src="https://user-images.githubusercontent.com/76904126/229498434-0cea4958-e38c-4946-a060-7dcbfbe9ea2a.png">
    
</details>


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
