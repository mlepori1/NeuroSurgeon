# Neurosurgeon

NeuroSurgeon is a package that enables researchers to uncover and manipulate circuits within models in Huggingface Transformers. NeuroSurgeon provides a simple API to instantiate Continuous Sparsification masks over linear, attention, and convolution layers in BERT, GPT, ResNet, and ViT-style models. In the future, more masking techniques will be included. Details on continuous sparsification can be found here: https://arxiv.org/abs/1912.04427

Examples of using NeuroSurgeon on a variety of models can be found in the `Models/test`. In particular, examples of training binary masks on image classification while keeping the underlying model frozen can be found in `Models/test/test_other_models.py`.

## Configuring `circuit_model`

To create a `circuit_model`, one must provide a `circuit_config` and a model from Huggingface Transformers. `circuit_config` takes in the following arguments.

- `mask_method: str` = "continuous_sparsification" : This should always be set to `continuous_sparsification`.
- `mask_hparams: dict = {}` : This contains hyperparameters that are specific to the masking technique specified by `mask_method`
- `target_layers: List[str] = []` : This contains a list of strings denoting the layers to mask. Can specify any number of layers, as long as they are implemented as nn.Linear, nn.Conv1d, nn.Conv2d, or transformers.pytorch_utils.Conv1D layers. Find the available layer names using `model.named_parameters()`
- `freeze_base: bool = True`: This boolean denotes whether the underlying weights and biases should be frozen during mask training or not. If this argument is set to true, then the underlying model will remain in eval mode, even if you call `circuit_model.eval()`. This changes the behavior of dropout and batchnorm layers, for example. If you're just trying to train binary masks, then set this value to true.
- `add_l0: bool = True`: This boolean denotes whether to add L0 regularization to any loss produced by the HF model's forward pass. If so, then L0 regularization will be applied to all binary masks, in accordance with continuous sparsification.
- `l0_lambda: float = 1E-8`: This parameter balances the L0 loss with any other loss returned by a HF forward pass. The default value works pretty well for cross entropy losses.

### Continuous Sparsification `mask_hparams`

- `ablation` : This hyperparameter controls the behavior of binary masks. It can take several values:
- - `none`: This value is used to train and evaluate the mask parameters. This is the default mode.
- - `zero_ablate`: This inverts a binary mask. This is useful for ablating a trained mask. Only use this after your mask has been trained!
- - `random_ablate`: Rather than zero-ing out the discovered circuit, you can replace it with randomly initialized values. This is largely equivalent to `zero_ablate`. Only use this after your mask has been trained!
- - `randomly_sampled`: This is useful for comparing the effect of ablating a discoverd circuit to ablating random circuits of equal size. It will randomly sample binary masks from the complement of the set of parameters in the discovered circuit, and ablate them. More precisely, if a trained mask contains parameters m in weight matrix M, then ablating it results in parameters m' from M. Let |m| = n. This parameter selects n parameters randomly from m' and ablates them.
- `mask_bias` : This is a boolean denoting whether to mask bias terms as well as weight terms. Typically, the pruning community does not mask biases.
- `mask_init_value` : A float determining how to initialize the masking parameters. 0.0 typically works well. More negative will result in smaller circuits, more positive will result in larger circuits.

## Practical Tips
Continuous sparsification relies on annealing a soft mask into a hard binary mask over a number of epochs. Empirically, 90 epochs and annealing according to an exponential schedule to a maximum value of 200 works well. All of these details can be found in the continuous sparsification paper. In order to implement this behavior, you'll need to define a callback in your training loop. Here is an example callback that I used previously. This callback is designed for use with the HF Trainer API. Unforunately, NeuroSurgeon does NOT yet support HF Trainer, so you'll need to inject this logic in a standard training loop!

This should be pretty straightforward (even simpler than the code below). After every epoch, you'll want to just set the temperature of the circuit_model to a new value, which you can calculate on the fly (see code below). It should just be this:
`circuit_model.temperature = new_temp`


Example Callback for computing new temperature at every epoch.
```
class TemperatureCallback(Callback):

    def __init__(self, total_epochs, final_temp, masks):
        self.l0_masks = masks
        self.temp_increase = final_temp**(1./total_epochs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.l0_masks["backbone"]:
            temp = pl_module.backbone.get_temp()
            pl_module.backbone.set_temp(temp * self.temp_increase)
        elif self.l0_masks["mlp"]:
            temp = pl_module.mlp.get_temp()
            pl_module.mlp.set_temp(temp * self.temp_increase)
```

## CRAFT + Circuits Idea
To find circuits in that implement CRAFT concepts, we need only to freeze the underlying model and define a loss function that rewards the model for representing one target concept, while penalizing it for representing any other concept. We will then only train binary masks to automatically discover disentangeled circuits that are responsible for the target concept. I recommend training with L0 regularization to get maximally sparse circuits. We will need to search over layer combinations to find the circuit, but starting one layer behind where the CRAFT concepts are calculated seems like a good start.

