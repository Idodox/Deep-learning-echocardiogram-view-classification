# Deep learning based echocardiogram view classification using spatio-temporal features

Masters thesis research in cooperation with prof. Dan Adam, Technion institute of technology, Israel.

## Introduction:

TL'DR: This project classifies echocardiogram videos to one of the 6 main classes without any user input by using a novel architecture which employs 3D CNN's.

Abstract: 
Through the years, improvements in medical imaging have enabled the collection of increasingly detailed and precise data useful for making diagnosis and monitoring treatments. In particular, Echocardiography has proven itself as an effective imaging tool in many cases [1]. It is thus routinely used in the diagnosis, management, and follow-up of patients with any suspected or known heart diseases. Analyzing echocardiogram data requires human interpretation by a trained professional, limiting the potential advancement of automatic diagnostic systems. This is partly due to relatively complex multi-view formats. Exacerbating the problem, several views may appear very similar to each other in different parts of the clip. Automatic view classification is thus an essential first step towards a fully autonomic diagnostic system

** Note: This code isn't well documented and could use a refactor. However, as this is a research project those will not be done at this time.

## Requirements
* Python 3
* [PyTorch](http://pytorch.org/)
* Torchvision
* Numpy
* PIL

### Not necessary but integrated into the code:
  * Comet
  * TQDM
  * Scikit-learn
  * matplotlib

## Preparation
* Download this code.
* Prepare data by converting all dicom videos to individual frames. The notebooks in this repo may help.

## Usage
* You will need to first update the location of the raw frames on your computer

* To make mini-clips (pickle) from each video's frames, use the python files in "dataset_creation"

* Once the clips for all classes are ready, update their location in run_state.py

* Use main.py for training a single model, and use main_crossval.py runs cross-validation. Note - State is handled by run_state.py


## Citation
If you use this code, please cite the following:
```
@repository{
  author={Ido Farhi},
  title={Deep learning based echocardiogram view classification using spatio-temporal features},
  year={2020}
}
```
