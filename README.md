# TrillView 

You can find an online example of TrillView in this link http://203.101.225.95

This repository contains scripts to visualise radial distributions of bioacoustic model predictions.

## Dataset
For our experiments we used the dataset [*BirdSet*]( https://huggingface.co/datasets/DBD-research-group/BirdSet) (Rauch et al. 2025)

You can find the lists of frames for our selected samples “**ssw-test-50**” and “**hsn-test-50**” under ``` \datasets```

## Generate labels
The script `birdset_labels.py` generates labels for the two BirdSet project components that we used in our experiment: “**ssw**” and “**hsn**”.

Note that these lists contain species groups that may need to be manually matched to the labels of the model used.

## Generate spatial distribution of classes
Use the `script generate_spatial_distribution.py` and a list of labels to generate a spatial distribution of the labels. For example: 

```python
python generate_spatial_distribution.py data/hsn_labels.csv radial
```
### parameters
```text
usage: generate_spatial_distribution.py label_file {radial,focal} 
           [-h] [-target_sps TARGET_SPS] [-target_sps_field TARGET_SPS_FIELD] [-radius RADIUS] 
           [-arc_pi_frac ARC_PI_FRAC] [-alphabetical {True,False}] 

Coordinate generation

positional arguments:
  label_file            Input path to label file
  {radial,focal}        distribution selection: radial or focal

optional arguments:
  -h, --help            show this help message and exit
  -target_sps           TARGET_SPS target species for focal distribution only. Use "" for binomials
  -target_sps_field     TARGET_SPS_FIELD field containing target species, default SCIENTIFIC_NAME
  -radius RADIUS        radius of circular distribution, default=1
  -arc_pi_frac          ARC_PI_FRAC fraction of pi angle for other classes in focal distribution, default=1
  -alphabetical         {True,False} If ordering the target_sps_field alphabetically or keep existing order, default=True

```

## Generate logits distribution
Use the spatial distribution generated for a matrix multiplication against the logits after softmax and masking of your model outputs. using something akin to:
```python
model = ...
model_labels_df = pd.read_csv(...)
# ensure there is a SCIENTIFIC_NAME

# load and match labels 
labels_df = pd.read_csv("/data/hsn_labels_matched.csv")
# ensure that species match with model labels
model_labels_df.reset_index(drop=False, inplace=True, names='label')
labels_merged = labels_df.merge(model_labels_df, on="SCIENTIFIC_NAME", how='left')
labels_idx = np.array(labels_merged['label'])

# load matching coordinates
coord_list = pd.read_csv("/data/hsn_labels_matched-radial-distrib-alphabetical.csv")
coord_list = coord_list.sort_values(by='index')
coord_list.index = coord_list['index']
import ast
target_coord = coord_list['coord'].apply(ast.literal_eval).tolist()
target_coord_arr = np.array(target_coord)
target_coord_arr_gpu = tf.Variable(target_coord_arr, dtype=tf.float32)
		/ torch.nn.Parameter(arget_coord_arr, device="cuda")

# process logits against target coordinates in model routine
logits = model(batch_frames)
confs = softmax(logits, axis=1)
confs_sel = gather(confs, labels_idx, axis=1)
output = confs_sel @ target_coord_arr_gpu
# save the batched output to a .csv file with input metadata as any other output
...
```

## Display in TrillView interactive viewer
Use `app.py` to display the results by running 
```python 
python app.py 
```
Then open the viewer on a browser, typically at http://127.0.0.1:8080/

Follow the instructions in the browser. You can use the example in 
```text
 /example/example_fig.csv 
 ```

If using docker
```
sudo docker build -t trillview .
sudo docker run -d -p 80:80 trillview
```

## Requirements
The code requires the libraries [plotly](https://plotly.com/), [dash](https://plotly.com/dash/), [Pillow](https://python-pillow.org/) as well as other common libraries.

## References and Sources

Bravo Sanchez, F. J., Woodbright, M. D., Hossain, M. R., English, N. B., Moore, S. T.  *Radial Distributions of Bioacoustic Model Predictions: A Visualisation Method that Aids Interpreting Results, Assessing Performance and Explaining Anomalies*. Manuscript under review at Ecological Informatics

Bravo Sanchez, F. J., English, N. B., Hossain, M. R., & Moore, S. T. *Improved analysis of deep bioacoustic embeddings through dimensionality reduction and interactive visualisation*. Ecological Informatics 2024. https://doi.org/10.1016/j.ecoinf.2024.102593 

Rauch, L. , Schwinger, R., Wirth, M. et al. 2025. *BirdSet: A Large-Scale Dataset for Audio Classification in Avian Bioacoustics*. [Paper presented at 13th International Conference on Learning Representations (ICLR 2025), Singapore. April.](https://proceedings.iclr.cc/paper_files/paper/2025/hash/484d254ff80e99d543159440a06db0de-Abstract-Conference.html)

