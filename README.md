# BPDSurv

## Requirements

The following packages are required to run the code. You can install them using `pip` or `conda`:

- python == 3.9.23 
- numpy == 1.26.4
- pandas == 2.3.1
- scikit-survival == 0.23.1
- torch == 2.4.1 
- torchvision == 0.19.1


## Data Preparation 

### WSIs

1. Download diagnostic WSIs from [TCGA](https://portal.gdc.cancer.gov/)
2. Use the WSI processing tool provided by [CLAM](https://github.com/mahmoodlab/CLAM) to extract [CONCH](https://github.com/mahmoodlab/CONCH) features, which we then save as `.pt` files for each WSI. 
The final structure of datasets should be as following:

```bash
DATA_ROOT_DIR/
    └──pt_files/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
```

`DATA_ROOT_DIR` is the base directory of cancer type (e.g. the directory to TCGA_BLCA).

### Genomics

In this work, we directly use the preprocessed genomic data provided by [MCAT](https://github.com/mahmoodlab/MCAT), stored in folder [dataset_csv](./dataset_csv).

## Training-Validation Splits

Splits for each cancer type are found in the `splits/5foldcv ` folder, which are randomly partitioned each dataset using 5-fold cross-validation. Each one contains splits_{k}.csv for k = 1 to 5. We follow the same splits as that of MOTCat.

## Quick Start

First, extract the graph features by running:

```bash
python models/extract_graph.py --h5_path <your_path> --pt_path <your_path> --graph_save_path <your_path>
```

Then, you can start training by running the bash script:

```bash
/bin/bash train.sh
```

## Acknowledgements

Huge thanks to the authors of following open-source projects:
[MCAT](https://github.com/mahmoodlab/MCAT) , 
[CLAM](https://github.com/mahmoodlab/CLAM) ,
[CONCH](https://github.com/mahmoodlab/CONCH) ,
[MOTCat](https://github.com/Innse/MOTCat) ,
[VLSA](https://github.com/liupei101/VLSA)
