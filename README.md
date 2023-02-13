# CoughSultant: COVID-19 detection from cough recordings
The following project was implemented by Ioannis Kaziales for the course [CS-577 (Machine Learning)](https://www.csd.uoc.gr/CSD/index.jsp?content=pg_courses_catalog&openmenu=demoAcc4&lang=en&course=156) (fall semester 2022-2023).

## Overview: 
The COVID-19 pandemic has had a profound impact on global health and the economy. Even though the disease has been largely contained, there is still a need for inexpensive, rapid, reliable, and non-invasive detection methods for identifying new cases. The development and use of such tools could be useful in mitigating the effect of the virus in our daily life. Among the COVID-19 cases, a dry cough is a common symptom and appears in approximately 67.7% of the cases. Analysis of speech and respiratory sounds with machine learning techniques has proven to be a promising approach to detecting respiratory diseases, including COVID-19, and can provide useful insights, enabling the design of diagnostic tools which are scalable, fast, and easily accessible to the general public. The aim of those methods is not to replace traditional diagnostic tools, but to complement them. Here, I will present the development process of "CoughSultant", a machine learning-based approach for the detection of COVID-19 from cough recordings.

## Acknowledgements
This projects uses the following resources:
- The [Coughvid](https://coughvid.epfl.ch/) dataset and automatic tools for cough detection and segmentation are used. Specifically, I used:
  - [dataset on Zenodo](https://doi.org/10.5281/zenodo.4048311)
  - [scripts on c4science](https://c4science.ch/diffusion/10770/)
  - [publication on Nature Scientific Data](https://doi.org/10.1038/s41597-021-00937-4)
- the [openSMILE python](https://audeering.github.io/opensmile-python/) library for extraction of high-level [eGeMAPSv02](https://sail.usc.edu/publications/files/eyben-preprinttaffc-2015.pdf) acoustic features from cough recordings
- the [Mlxtend](https://rasbt.github.io/mlxtend/) library for feature selection and plotting

## How to use the project: 
Firstly, create a virtual environment  (I used python version `3.9.16`) and install the necessary Python library dependencies. You can do this by running the following command:
```bash
pip install -r requirements.txt
```
Download the [COUGHVID dataset](https://zenodo.org/record/7024894) and the [COUGHVID scripts](https://c4science.ch/diffusion/10770/) using:
```bash
wget https://zenodo.org/record/7024894/files/public_dataset_v3.zip
git clone https://c4science.ch/diffusion/10770/coughvid.git
```

COUGHVID's source code includes some useful automatic tools that were used in this project. Specifically:
- [`convert_files.py`](https://c4science.ch/diffusion/10770/browse/master/src/convert_files.py) is a script that converts all the compressed `.webm` and `.ogg` files in the COUGHVID dataset to `.wav ` format. 

  > Note: you must have FFMPEG installed for this to work. 

- [`segmentation.py`](https://c4science.ch/diffusion/10770/browse/master/src/segmentation.py) contains the function `segment_cough()` which segments each file into individual coughs

- [`DSP.py`](https://c4science.ch/diffusion/10770/browse/master/src/DSP.py) contains the function `classify_cough()` which uses a trained XGB model to classify whether or not a given recording contains cough sounds.

After unzipping the dataset, you will notice that the recordings are in `.webm` and `.ogg` files. You can use COUGHVID's `convert_files.py` to convert them to `.wav` format. The following script also checks that the conversion was successful and removes the other formats:

```python
import os
import glob
from src.convert_files import convert_files

dir = './coughvid_20211012/' # dataset directory
convert_files(dir)

# check that all files were converted
ogg_files = glob.glob(os.path.join(dir, '*.ogg'))
webm_files = glob.glob(os.path.join(dir, '*.webm'))
wav_files = glob.glob(os.path.join(dir, '*.wav'))
print(f".wav files: {len(wav_files)}, .ogg and .webm_files: {len(ogg_files) + len(webm_files)}")

# remove the .ogg and .webm_files
for file in wav_files:
    f = file.split('.wav')[0]
    if f+'.ogg' in ogg_files:
    	os.remove(f+'.ogg')
        print(f"removed {f}.ogg")
	elif f+'.webm' in webm_files:
    	os.remove(f+'.webm')
        print(f"removed {f}.webm")
```

---

Then, you can use the `data_exploration.ipynb` notebook to get familiar with the dataset, visualize the distributions and perform pre-processing. After running this script, a new dataset (`dataset.csv`) will be generated. This dataset contains `15,293` entries out of the `34,434` entries of the original dataset (the rest had missing labels $\to$ not suitable for supervised machine learning approaches, or bad-quality recordings). The features contain only a small subset of features from the original dataset, but high-level acoustic features are extracted from the recordings and added to the new dataset.



Finally, the `classification_AUC.ipynb` notebook performs multi-class classification ('healthy', 'COVID-19', 'symptomatic') using machine learning techniques. It loads the dataset and splits it to 85% train set and 15% test set. It performs forward feature selection to approximate the best subset of features. Then it selects the best model configuration using grid search on the train set, it approximates the performance of the best configuration (trained on the whole training set) with the test set and returns the final model, trained on all the data.



The previous notebook uses `weighted One-vs-Rest ROC-AUC` as a scoring function for feature selection and hyperparameter tuning. The `classification_F1.ipynb` notebook performs the same procedure/pipeline. The only difference is that `weighted F1` is used as a scoring function. This results in better predictions (less affected by class imbalance).



## Future Work:

Some ideas I would like to further explore:

- perform data augmentation to improve the huge class imbalance in the dataset.  This can be achieved using the [audiomentations](https://pypi.org/project/audiomentations/) library .
- use a neural network architecture instead of classic machine learning approaches. A good example is [the winner of the  DiCOVA 2021 Challenge](https://dicova2021.github.io/docs/reports/team_Brogrammers_DiCOVA_2021_Challenge_System_Report.pdf)

 
