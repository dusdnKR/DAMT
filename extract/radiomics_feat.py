import os
import numpy as np
import csv
import SimpleITK as sitk
import radiomics
import logging
import scipy.stats as stats
from collections import OrderedDict
from tqdm import tqdm

# set level for all classes
logger = logging.getLogger("radiomics")
logger.setLevel(logging.ERROR)
# ... or set level for specific class
logger = logging.getLogger("radiomics.glcm")
logger.setLevel(logging.ERROR)

def textureFeaturesExtractor(img, roi, roiNum):
    roi_name = ["", "GM", "WM", "CSF"]
    textureFeaturesDict = {}

    glcmFeatures = radiomics.glcm.RadiomicsGLCM(img, roi, label=roiNum, binCount=128, verbose=True, interpolator=None, symmetricalGLCM=True)
    glcmFeatures._initCalculation()
    glcmFeatures._calculateMatrix()
    glcmFeatures._calculateCoefficients()
    
    textureFeaturesDict['GLCM_Autocorrelation_' + roi_name[roiNum]] = glcmFeatures.getAutocorrelationFeatureValue().item()
    textureFeaturesDict['GLCM_ClusterTendency_' + roi_name[roiNum]] = glcmFeatures.getClusterTendencyFeatureValue().item()
    textureFeaturesDict['GLCM_Contrast_' + roi_name[roiNum]] = glcmFeatures.getContrastFeatureValue().item()
    textureFeaturesDict['GLCM_Correlation_' + roi_name[roiNum]] = glcmFeatures.getCorrelationFeatureValue().item()
    textureFeaturesDict['GLCM_DifferenceAverage_' + roi_name[roiNum]] = glcmFeatures.getDifferenceAverageFeatureValue().item()
    textureFeaturesDict['GLCM_DifferenceEntropy_' + roi_name[roiNum]] = glcmFeatures.getDifferenceEntropyFeatureValue().item()
    textureFeaturesDict['GLCM_DifferenceVariance_' + roi_name[roiNum]] = glcmFeatures.getDifferenceVarianceFeatureValue().item()
    textureFeaturesDict['GLCM_JointEnergy_' + roi_name[roiNum]] = glcmFeatures.getJointEnergyFeatureValue().item()
    textureFeaturesDict['GLCM_JointEntropy_' + roi_name[roiNum]] = glcmFeatures.getJointEntropyFeatureValue().item()
    textureFeaturesDict['GLCM_IMC1_' + roi_name[roiNum]] = glcmFeatures.getImc1FeatureValue().item()
    textureFeaturesDict['GLCM_IMC2_' + roi_name[roiNum]] = glcmFeatures.getImc2FeatureValue().item()
    textureFeaturesDict['GLCM_IDM_' + roi_name[roiNum]] = glcmFeatures.getIdmFeatureValue().item()
    textureFeaturesDict['GLCM_MCC_' + roi_name[roiNum]] = glcmFeatures.getMCCFeatureValue().item()
    textureFeaturesDict['GLCM_IDMN_' + roi_name[roiNum]] = glcmFeatures.getIdmnFeatureValue().item()
    textureFeaturesDict['GLCM_InverseDifference_' + roi_name[roiNum]] = glcmFeatures.getIdFeatureValue().item()
    textureFeaturesDict['GLCM_InverseVariance_' + roi_name[roiNum]] = glcmFeatures.getInverseVarianceFeatureValue().item()
    textureFeaturesDict['GLCM_MaximumProbability_' + roi_name[roiNum]] = glcmFeatures.getMaximumProbabilityFeatureValue().item()
    textureFeaturesDict['GLCM_SumAverage_' + roi_name[roiNum]] = glcmFeatures.getSumAverageFeatureValue().item()
    textureFeaturesDict['GLCM_SumEntropy_' + roi_name[roiNum]] = glcmFeatures.getSumEntropyFeatureValue().item()
    textureFeaturesDict['GLCM_SumofSquares_' + roi_name[roiNum]] = glcmFeatures.getSumSquaresFeatureValue().item()

    glszmFeatures = radiomics.glszm.RadiomicsGLSZM(img, roi, label=roiNum, binCount=128, verbose=True, interpolator=None)
    glszmFeatures._initCalculation()
    glszmFeatures._calculateMatrix
    glszmFeatures._calculateCoefficients()

    textureFeaturesDict['GLSZM_LargeAreaEmphasis_' + roi_name[roiNum]] = glszmFeatures.getLargeAreaEmphasisFeatureValue().item()
    textureFeaturesDict['GLSZM_GLNN_' + roi_name[roiNum]] = glszmFeatures.getGrayLevelNonUniformityNormalizedFeatureValue().item()
    textureFeaturesDict['GLSZM_SZNN_' + roi_name[roiNum]] = glszmFeatures.getSizeZoneNonUniformityNormalizedFeatureValue().item()
    textureFeaturesDict['GLSZM_ZoneEntropy_' + roi_name[roiNum]] = glszmFeatures.getZoneEntropyFeatureValue().item()
    
    return textureFeaturesDict

def main():
    data_path = "/NFS/Users/kimyw/data/sample"
    textures = []
    for subject in tqdm(os.listdir(data_path)):
        sub_path = os.path.join(data_path, subject, "mri")
        if not os.path.exists(os.path.join(sub_path, "gmwmcsf.nii.gz")):
            continue
        img = sitk.ReadImage(os.path.join(sub_path, "brainmask.nii.gz"))
        roi = sitk.ReadImage(os.path.join(sub_path, "gmwmcsf.nii.gz"))
        texture_dict = {'subject': subject}
        texture_dict.update(textureFeaturesExtractor(img, roi, 1))
        texture_dict.update(textureFeaturesExtractor(img, roi, 2))
        texture_dict.update(textureFeaturesExtractor(img, roi, 3))
        textures.append(texture_dict)

    with open(os.path.join(data_path, "radiomics_texture.csv"), "w") as f:
        writer = csv.DictWriter(f, textures[0].keys())
        writer.writeheader()
        writer.writerows(textures)
    


if __name__ == "__main__":
    main()