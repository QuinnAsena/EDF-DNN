# Glossary

Machine-learning terms in plain words, with where they show up in this repo
and the nearest R or statistics idea. Add a row whenever a new term comes up.

## Data and samples

| Term | Plain meaning | In our code | R / stats analogy |
|---|---|---|---|
| Tensor | PyTorch's array: numbers in any number of dimensions, stored on CPU or GPU | everything that goes into a model | `array()`, `matrix()` |
| Sample | One example the model learns from: a pixel (MLP) or a patch (CNN) | `data/datasets.py` | one row of a data frame |
| Feature / predictor | One input variable, e.g. `tcc` | `variables:` in `configs/data.yaml` | covariate, explanatory variable |
| Label / target | The answer the model should predict | `burned01` | response variable `y` |
| Dataset | Object that answers "how many samples?" and "give me sample i" | `data/datasets.py` | |
| DataLoader | Draws shuffled mini-batches from a Dataset | `train/loop.py` | |
| Batch / mini-batch | A handful of samples processed together before one weight update. Full-batch = all data at once, as in the old code | `batch_size` in experiment configs | |
| Epoch | One full pass through the training data | `max_epochs` | one iteration over all rows |
| Class imbalance | Far more of one class than the other (here about 6 unburned per burned pixel) | | rare-event data |
| Downsampling | Drop majority-class samples from the training folds until classes are balanced | `imbalance: downsample` | `sample()` on the unburned rows |
| `pos_weight` | Alternative to downsampling: keep every sample, but count each burned pixel's error (unburned / burned) times more in the loss | `imbalance: pos_weight` | `weights =` in `glm()` |
| Scaling / normalisation | Rescale each variable to a similar range (0-1, or mean 0 and sd 1) so no variable dominates just by its units | `data/normalise.py` | `scale()` |
| Min-max scaling | (x - min) / (max - min), with min and max from the training pixels, so those span 0-1. Squashed by extreme values: heavily skewed variables need a log first | `MinMaxScaler`, saved by `save_scaler` | `scales::rescale()` |
| Pickle | Python's way of saving any object to a file and loading it back unchanged | `scaler.pkl` | `saveRDS()` / `readRDS()` |
| float32 / float64 | Single (about 7 digits) and double (about 16 digits) precision numbers. Networks compute in float32: half the memory, ample precision | rasters are float32 | R numerics are always float64 |
| Data leakage | Information from validation or test data getting into training (e.g. a scaler fitted on all data), which makes scores look better than they are | why folds and scaling sit in `data/` | |

## Network building blocks

| Term | Plain meaning | In our code | R / stats analogy |
|---|---|---|---|
| Neuron (unit) | Takes a weighted sum of its inputs plus a bias, then applies an activation function | | one linear predictor |
| Layer, `nn.Linear` | A set of neurons. "Linear" or "fully connected": every input feeds every neuron | `fc1`, `fc2`, `fc3` in `models/mlp.py` | `X %*% W + b` |
| Hidden layer, `hidden_size` | Layers between input and output; `hidden_size` is the number of neurons in each | `hidden_size` | |
| Weights and biases (parameters) | The numbers the network learns | | regression coefficients |
| Activation function, ReLU | A bend applied after each layer. ReLU = max(0, x). Without it, stacked layers collapse into one linear model | `nn.ReLU` in `models/mlp.py` | `pmax(0, x)` |
| MLP (multilayer perceptron) | Stacked linear layers with activations between them. An MLP with no hidden layer is logistic regression | `models/mlp.py` | `glm(family = binomial)` plus learned non-linear transforms of the predictors |
| Logit | The raw model output, from minus to plus infinity: the log-odds | every model returns logits | `predict(type = "link")` |
| Sigmoid | Turns a logit into a probability between 0 and 1 | applied at evaluation and prediction | `plogis()` |
| Dropout | During training, randomly switch off a fraction of neurons on each pass, so the network can't rely on any single one. Reduces overfitting. Off at prediction time | `dropout` | |
| Bagging | Train many separate models on bootstrap resamples and average them. Dropout is often described as a cheap version of bagging many thinned networks inside one network | not used | random forest = bagged trees |
| `model.train()` / `model.eval()` | Switch the model into training mode (dropout on) or evaluation mode (dropout off) | `train/loop.py` | |

## Training

| Term | Plain meaning | In our code | R / stats analogy |
|---|---|---|---|
| Loss function | One number measuring how wrong the predictions are; training makes it smaller | | negative log-likelihood |
| `BCEWithLogitsLoss` | Binary cross-entropy computed straight from logits (the sigmoid is applied inside, which is numerically safer) | `train/loop.py` | binomial deviance / (2n) |
| Gradient | For each weight: which direction, and how strongly, to change it to reduce the loss | | partial derivatives |
| Backpropagation, `loss.backward()` | Computes the gradients for all weights at once, layer by layer, using the chain rule | `train/loop.py` | |
| `optimizer.zero_grad()` | Clears the previous gradients; PyTorch adds them up otherwise | `train/loop.py` | |
| Optimizer, Adam | Uses the gradients to update the weights (`optimizer.step()`). Adam adapts the step size for each weight | `train/loop.py` | `optim()`, but in many small steps on mini-batches |
| Learning rate | Size of each update step. Too big: training jumps around; too small: slow | `learning_rate` | step size |
| Overfitting | The model memorises the training data and does worse on new data. Seen when validation loss rises while training loss keeps falling | | |
| Early stopping, patience | Stop when validation loss hasn't improved for `patience` epochs, and keep the best weights | `patience` | |
| Hyperparameter | A setting we choose rather than the model learning it: hidden size, dropout, learning rate, batch size | `configs/experiments/` | tuning parameter |
| Grid search | Try every combination of hyperparameter values | lists in experiment configs | `expand.grid()` + a loop |
| Seed | Fixed starting point for random numbers, so runs are reproducible | `1984` | `set.seed()` |
| Checkpoint, `state_dict` | The saved weights of a trained model (`.pt` file) | `runs/<experiment>/` | `saveRDS(model)` |
| `torch.no_grad()` | Switches off gradient tracking when only predicting: faster, less memory | `train/loop.py` | |
| Device, GPU, CUDA | Where tensors and computations live. GPU = graphics card, much faster for networks; CUDA = NVIDIA's GPU platform | `device: auto` | |

## Validation and evaluation

| Term | Plain meaning | In our code | R / stats analogy |
|---|---|---|---|
| Training / validation / test set | Fit on training; pick hyperparameters and stop early on validation; report once on test | `train/run.py` | |
| Cross-validation, fold, k | Split the data into k folds; each fold takes a turn as validation | `n_folds` | `rsample::vfold_cv()`, `caret` |
| Stratified | Every fold keeps the same burned/unburned ratio | | `strata =` |
| Spatial block CV | Folds made of whole geographic blocks, so neighbouring pixels can't leak information across (Roberts et al. 2017) | `data/blocks.py`, `block_size_km` | `blockCV` package |
| Threshold | Probability cut-off for calling a pixel burned (default 0.5) | `train/metrics.py` | |
| Confusion matrix | Counts of true/false positives and negatives at a threshold | `train/metrics.py` | `table(observed, predicted)` |
| Precision | Of the pixels predicted burned, the fraction that actually burned: TP / (TP + FP) | `train/metrics.py` | positive predictive value |
| Recall (sensitivity) | Of the pixels that actually burned, the fraction the model caught: TP / (TP + FN) | `train/metrics.py` | sensitivity |
| Specificity | Of the pixels that did not burn, the fraction correctly left out: TN / (TN + FP) | `train/metrics.py` | true negative rate |
| True skill statistic (TSS) | Recall + specificity - 1, at a threshold. 0 = random, 1 = perfect; unaffected by the burned share (Allouche et al. 2006) | `train/metrics.py` | Youden's J |
| Burned share (prevalence) | Fraction of the evaluated pixels that burned; the no-skill baseline for average precision and the Brier skill score | `train/metrics.py` | `mean(y)` |
| ROC AUC | Chance the model ranks a random burned pixel above a random unburned one. 0.5 = coin toss, 1 = perfect | `train/metrics.py` | `pROC::auc()` |
| Average precision (PR AUC) | Summary of the precision-recall curve; more informative when classes are imbalanced. A useless model scores the burned share (about 0.14 here) | `train/metrics.py` | `PRROC` |
| Brier score | Mean squared difference between predicted probability and the 0/1 outcome; lower is better (Brier 1950) | `train/metrics.py` | `mean((p - y)^2)` |
| Brier skill score | Brier score relative to always predicting the burned share. 1 = perfect, 0 = no skill, below 0 = worse than that | `train/metrics.py` | `1 - BS / BS_reference` |
| Log loss | Binary cross-entropy as an evaluation score: like Brier, but punishes confident mistakes much harder; lower is better | `train/metrics.py` | binomial deviance / (2n) |
| Calibration | Whether a predicted 0.7 really means about 70% of such pixels burn | | |

## Spatial models (CNN, U-Net)

| Term | Plain meaning | In our code | R / stats analogy |
|---|---|---|---|
| Patch | A square window of pixels (variables x height x width) around a location; the input of the CNN | `data/datasets.py` | `terra::crop()` around one cell |
| Channel | One layer of the input stack; our variables are the input channels | | one raster layer |
| Convolution, kernel / filter | Slide a small window of learned weights across the map; each output pixel is a weighted sum of its neighbourhood | `models/cnn.py` | `terra::focal(r, w = matrix)`, with `w` learned |
| Feature map | The output of a convolution: a derived raster the network learned to compute | | a derived layer |
| Pooling | Shrink a map by summarising each small window (e.g. its maximum): coarser detail, wider context | `models/cnn.py` | `terra::aggregate(r, fact = 2, fun = "max")` |
| Receptive field | How far around a pixel the network can "see" | | focal window size |
| CNN (convolutional neural network) | Stacks of convolution and pooling layers, then a small fully connected head | `models/cnn.py` | |
| U-Net | Encoder (convolutions + pooling: shrinks the map, gains context) then decoder (upsampling back to full resolution); predicts every pixel of a tile at once | `models/unet.py` | |
| Upsampling | The reverse of pooling: back to a finer resolution | `models/unet.py` | `terra::disagg()` |
| Skip connection | Passes fine detail from the encoder straight to the decoder at the same resolution | `models/unet.py` | |
