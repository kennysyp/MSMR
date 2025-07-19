# MSMR
  Code for Multi-Scale Multimodal Representation for Enhanced Survival Prediction in Computational Pathology.

## Data preparation

- **Data Download**

  To obtain pathology image data, please refer to the [GDC Data Portal](https://portal.gdc.cancer.gov/) and [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/) to download the corresponding diagnostic WSI.

  For molecular characterization data, please refer to [cBioPortal](https://www.cbioportal.org/) and [Genome Data Commons Portal](https://portal.gdc.cancer.gov/).

  For biological pathway data, please refer to [MSigDB](https://www.gsea-msigdb.org/gsea/index.jsp).

- **Pathological image data processing**
  
  First, WSI was downsampled and converted to HSV color space, and the tissue region was extracted using saturation threshold + morphological operation. The image patches corresponding to the pathological images at x40 and x20 magnification were cropped from the tissue region without overlap. Subsequently, the pre-trained [UNI](https://github.com/mahmoodlab/UNI) was used for feature extraction to extract the 1024-dimensional feature vector of each patch. All WSI preprocessing was completed using the [CLAM](https://github.com/mahmoodlab/CLAM) tool.

### Data Partitioning-Training-Validation

- **Data Partitioning**
  
  To evaluate the performance of the algorithm, we used five-fold cross validation to conduct experiments and stratified by the sampling location of the histological sections to divide each data set. Please refer to this article for why the data should be divided according to the sampling location instead of randomly dividing it according to a specific ratio: [The impact of site-specific digital histology signatures on deep learning model accuracy and bias](https://www.nature.com/articles/s41467-021-24698-1).

- **Training-Validation**
 
  Training：
  ```
  ./brca.sh 0
  ```
  Validation:
  ```
  ./brca_surv.sh 0 msmr
  ```
