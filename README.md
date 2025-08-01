# MSPPDepth

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Multi-Scale Planarity Prior based Self-Supervised Monocular Depth Estimation**

## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
pip3 install mmcv-full==1.3.0 mmsegmentation==0.11.0  
pip install timm einops IPython
```
We ran our experiments with PyTorch 1.9.0, CUDA 11.1, Python 3.7 and Ubuntu 18.04.


## <span id="datasets">💾 Preparing datasets</span>
### KITTI
For KITTI dataset, you can prepare them as done in [Monodepth2](https://github.com/nianticlabs/monodepth2). Note that we directly train with the raw png images and do not convert them to jpgs. You also need to generate the groundtruth depth maps before training since the code will evaluate after each epoch. For the raw KITTI groundtruth (`eigen` eval split), run the following command. This will generate `gt_depths.npz` file in the folder `splits/kitti/eigen/`.
```shell
python export_gt_depth.py --data_path /home/datasets/kitti_raw_data --split eigen
```
For the improved KITTI groundtruth (`eigen_benchmark` eval split), please directly download it in this [link](https://www.dropbox.com/scl/fi/dg7eskv5ztgdyp4ippqoa/gt_depths.npz?rlkey=qb39aajkbhmnod71rm32136ry&dl=0). And then move the downloaded file (`gt_depths.npz`) to the folder `splits/kitti/eigen_benchmark/`.


### Cityscapes
For Cityscapes dataset, we follow the instructions in [ManyDepth](https://github.com/nianticlabs/manydepth). First Download `leftImg8bit_sequence_trainvaltest.zip` and `camera_trainvaltest.zip` in its [website](https://www.cityscapes-dataset.com/), and unzip them into a folder `/path/to/cityscapes/`. Then preprocess CityScapes dataset using the followimg command:
```shell
python prepare_cityscapes.py \
--img_height 512 \
--img_width 1024 \
--dataset_dir /path/to/cityscapes \
--dump_root /path/to/cityscapes_preprocessed \
--seq_length 3 \
--num_threads 8
```
Remember to modify `--dataset_dir` and `--dump_root` to your own path. The ground truth depth files are provided by ManyDepth in this [link](https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip), which were converted from pixel disparities using intrinsics and the known baseline. Download it and unzip into `splits/cityscapes/`

**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## ⏳ Training

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.


**Monocular training:**
```shell
python train.py --model_name mono_model --data_path path/to/your/datasets/folder --learning_rate 5e-5 
```

### GPUs

The code can only be run on a single GPU.
You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable:
```shell
CUDA_VISIBLE_DEVICES=1 python train.py --model_name mono_model --data_path path/to/your/datasets/folder --learning_rate 5e-5
```

## 📊 Evaluation

🔹 KITTI

To prepare the ground truth depth maps, run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```

Assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.
To evaluate a model on KITTI, run:
```shell
python evaluate_depth.py --load_weights_folder path/to/your/weights/folder --eval_mono
```


🔹 Cityscapes

Download cityscapes depth ground truth (provided by manydepth) for evaluation:
```bash
cd splits/cityscapes/
wget https://storage.googleapis.com/niantic-lon-static/research/manydepth/gt_depths_cityscapes.zip
unzip gt_depths_cityscapes.zip
```
To evaluate a model on Cityscapes, run:
```
python evaluate_cs_depth.py \
--load_weights_folder path/to/your/weights/folder \
--cityscapes_path path/to/your/datasets/folder
```

🔹 NYU

```
python evaluate_nyu_depth.py --load_weights_folder path/to/your/weights/folder --eval_mono
```


## 🖼️ Prediction for a single image

You can predict scaled disparity for a single image with:

```shell
python test_simple.py --image_path path/to/your/test/test_image.jpg --model_path path/to/your/models/folder
```

## Acknowledgement
Thanks the authors for their works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[MonoViT](https://github.com/zxcqlf/monovit)
