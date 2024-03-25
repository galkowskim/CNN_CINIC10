# CNN_CINIC10

Authors:
- [Mikołaj Gałkowski](https://www.github.com/galkowskim)
- [Hubert Bujakowski](https://www.github.com/hbujakow)

This project was conducted as part of the "Deep Learning" course at the Warsaw University of Technology. The objective was to develop a convolutional neural network capable of classifying images from the [CINIC-10 dataset](https://www.kaggle.com/datasets/mengcius/cinic10). This dataset comprises 270,000 images, each sized at 32x32 pixels and categorized into 10 classes. Derived from the CIFAR-10 dataset, CINIC-10 retains the same classes but with resized images. The dataset is partitioned into three subsets: training, validation, and test, containing 90,000 images each.

# Training

1. Download dataset and unzip it in the `data` folder.

2. Create environment using conda:

```bash
conda env create -f environment.yml
```

3. Go into the `src` directory.

4. Modify `config.json` to your needs and run `python train_model.py` to train the model (look into the argparse options for path details).

```python
python train_model.py --config <path_to_config_file> --checkpoints <path_to_save_checkpoints>
```

# Loading the model from checkpoint (after training)
Note: we have not released the checkpoint files, so you will have to train the model yourself to get the checkpoint files due to the large size of them.

```python
# Load the model
checkpoint_path = "path/to/checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
# If you want to continue training from where you left off, you can also load the optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

# Comitee

`Comitee` folder enables you to create majority voting ensemble of models. You need to place checkpoints of models you want to use inside the `Comitee/augmented` or `Comitee/non_augmented` folder. Then you can run the notebook `Comitee/comitee.ipynb` to create the ensemble (modify to your needs).


# Results

You can find the results of our experiments in the "results" folder, where we have aggregated the outcomes of multiple runs conducted with different seeds. Additionally, the "checkpoints" folder contains the checkpoints from our experiments. To reproduce our results, you can utilize the provided configuration file available in the same directory.


| Model Name                        | Epochs | Aug. | lr    | Avg. accuracy          |
|-----------------------------------|--------|------|-------|------------------------|
| CustomCNN                         | 30     |  -   | 0.01  | 0.6482 ± 0.0045        |
| CustomCNN                         | 30     |  ✓   | 0.01  | 0.6897 ± 0.0044        |
| ResNetBasedModelFor32x32Images    | 30     |  -   | 0.1   | **0.6853 ± 0.0090**    |
| ResNetBasedModelFor32x32Images    | 30     |  ✓   | 0.1   | 0.7632 ± 0.0017        |
| VGG16BasedModelFor32x32Images     | 30     |  -   | 0.01  | 0.6823 ± 0.0043        |
| VGG16BasedModelFor32x32Images     | 30     |  ✓   | 0.01  | 0.7866 ± 0.0002        |
| WideModel                         | 75     |  -   | 0.01  | 0.6786 ± 0.0134        |
| WideModel                         | 75     |  ✓   | 0.01  | **0.7945 ± 0.0044**    |
| PretrainedAlexNet                 | 30     |  -   | 0.001 | 0.8208 ± 0.0021        |
| PretrainedAlexNet                 | 30     |  ✓   | 0.001 | 0.8121 ± 0.0005        |
| PretrainedResNet                  | 15     |  -   | 0.001 | 0.8550 ± 0.0011        |
| PretrainedResNet                  | 20     |  ✓   | 0.001 | **0.8655 ± 0.0010**    |
| PretrainedVGG16                   | 15     |  -   | 0.001 | 0.8430 ± 0.0017        |
| PretrainedVGG16                   | 15     |  ✓   | 0.001 | 0.8433 ± 0.0003        |
