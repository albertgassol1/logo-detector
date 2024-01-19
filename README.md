# logo-detector

This repo implements logo detection using the [Flick-27](http://image.ntua.gr/iva/datasets/flickr_logos/) dataset. The available models are FasterRCNN and SSD from [torchvision](http://image.ntua.gr/iva/datasets/flickr_logos/). By default, the pre-trained models are loaded before training.


## Installation
This project was developed and tested using a conda environment. To install the environent run the following commands:


```
conda env create -f environment.yaml
conda activate detection
pip install -r requirements.txt
```

Note: for a faster environment installation, consider installing ```mamba``` to replace ```conda```.

## Data
The dataset is automatically downloaded, parsed and splitted into train, validation and test splits by adding the flag ```--download``` when running the trainer.

The data for each split is randomly selected and the percentage of data for each split can be setted in the config file. By default, the splits are created when the data is downloaded, but one can regenerate a new set of splits by adding ```--compute_splits``` when running the trainer.

## Train

To train a model run the training script:
```
python train.py --config <relative_path_to_config_file>
```

The trainer incudes the following additional arguments:
```
--dataset-path [ralative_path_to_dataset]: path where the data will be downloaded. By default it is stored in ./data
--download: flag to download the data. Only needed when the trainer is first run. Once the data is downloaded, it is no longer needed.
--compute_splits: flag to recompute the train, validation and test splits.
--gpu [auto (default), cpu]: whether to use gpu or cpu.
--resume [reative_path_to_checpoint]: if provided, the training resumes from the given checkpoint.
--output-dir [relative_path_to_output_dir]: directory where the checkpoints are stored.
```

Examples:
- FasterRCNN:
    ```
    python train.py --config config/frcnn.yaml --download
    ```
- SSD:
    ```
    python train --config config/ssd.yaml
    ```
Note: the trainer uses ```wandb``` (Weights & Biases) for logging and visualization. One must create and account to access the logging.

## Config files
The config files are located in the folder ```config```. They contain relevant parameters for:
1. Data loading and processing (e.g. wheather to use data augmentation, image resizing, batch size)
2. Training (e.g. optimizer, hyperparameters)
3. Choice of model and wheater to freeze the backbone to do transfer-learning (otherwise the model is simply fine-tuned).
4. Choice of metrics for evaluation and testing.


## Inference
A trained FasterRCNN model can be downloaded [here](https://drive.google.com/file/d/1c-N008jM6fUR1zWdoTu5wxYPkKEVNNmc/view?usp=sharing). 

This project provides an inferencing script which also has the option to compute metrics (IoU and mAP) if ground truth labels are provided.

To run the inferencing:

```
python inference.py --config [relative_path_to_same_config_used_in_training] --model_checkpoint [relative_path_to_checkpoint] --test_folder [relative_path_to_test_images_folder]
```

The script provides the following optional arguments:
```
--output_path [relative_output_path]: Output directory where predictions (and metrics) are stored. The default is ./inference_output.
--eval_metrics: Flag to compute metrics. In orther to work, a file named test.json must be present in test_folder. It follows the same structure as the json files computed by the dataset parser. A test.json file is automatically computed by the training script when the dataset is downloaded.
--detection_th [float]: probability threshold used to accept a detection. Default is 0.7.
--gpu [auto (default), cpu]: whether to use gpu or cpu.
```

If the evaluation is enabled, the ground truth images with the bounding boxes are also stored in the output directory. Additionally, the metrics are stores in a json file in the output directory.

Example using the provided trained model (store it in folder ```./checkpoints```):
```
python inference.py --config config/frcnn.yaml --model_checkpoint checkpoints/FRCNN_v1_latest.pth --test_folder data/flickr_logos_27_dataset/test --eval_metrics
```

Reminder: if you have not used the trainer first and you want to test the model on the dataset, run the trainer once to download and parse the dataset with the flag ```--download```. After the dataset is downloaded and parsed, you can stop the training.

## TODOs:
- Implement other methods (YOLO and DetectronV2)
- Implement an option to add a triplet loss to enhance the logo classification.
- Implement test-time augmentation: Add rotations/cropping of the same image. Inference all variants and average results (classification and boxes location)
- Make use of more data: there are images which only show a logo which are not used. A portion of these can be added to the dataloader. Additionally, these logos could be pasted to random images (street, scene, etc) to have more data.
