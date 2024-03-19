# CNN_CINIC10

# Training

Modify `config.json` and run `python train_model.py` to train the model (look into the argparse options for path details).

```python
python train_model.py --config ... --checkpoints ...
```

# Loading the model from checkpoint

```python
# Load the model
checkpoint_path = "path/to/checkpoint.pth"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
# If you want to continue training from where you left off, you can also load the optimizer state
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```