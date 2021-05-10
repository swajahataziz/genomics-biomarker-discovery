

import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import time, pickle
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


# tumor abbreviations map
TCGA_MAP = {
    "acute myeloid leukemia": "LAML",
    "adrenocortical cancer": "ACC",
    "bladder urothelial carcinoma": "BLCA",
    "brain lower grade glioma": "LGG",
    "breast invasive carcinoma": "BRCA",
    "cervical & endocervical cancer": "CESC",
    "cholangiocarcinoma": "CHOL",
    "colon adenocarcinoma": "COAD",
    "diffuse large B-cell lymphoma": "DLBC",
    "esophageal carcinoma": "ESCA",
    "glioblastoma multiforme": "GBM",
    "head & neck squamous cell carcinoma": "HNSC",
    "kidney chromophobe": "KICH",
    "kidney clear cell carcinoma": "KIRC",
    "kidney papillary cell carcinoma": "KIRP",
    "liver hepatocellular carcinoma": "LIHC",
    "lung adenocarcinoma": "LUAD",
    "lung squamous cell carcinoma": "LUSC",
    "mesothelioma": "MESO",
    "ovarian serous cystadenocarcinoma": "OV",
    "pancreatic adenocarcinoma": "PAAD",
    "pheochromocytoma & paraganglioma": "PCPG",
    "prostate adenocarcinoma": "PRAD",
    "rectum adenocarcinoma": "READ",
    "sarcoma": "SARC",
    "skin cutaneous melanoma": "SKCM",
    "stomach adenocarcinoma": "STAD",
    "testicular germ cell tumor": "TGCT",
    "thymoma": "THYM",
    "thyroid carcinoma": "THCA",
    "uterine carcinosarcoma": "UCS",
    "uterine corpus endometrioid carcinoma": "UCEC",
    "uveal melanoma": "UVM",
}

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-test-split-ratio', type=float, default=0.3)
    args, _ = parser.parse_known_args()
    
    print('Received arguments {}'.format(args))

    tcga_input_path = os.path.join('/opt/ml/processing/input', 'TCGApancanRNA')

    tcga_annot_input_path = os.path.join('/opt/ml/processing/input', 'TCGAannot')
    
    gtex_input_path = os.path.join('/opt/ml/processing/input', 'GTExdata')

    gtex_annot_input_path_1 = os.path.join('/opt/ml/processing/input', 'E-MTAB-2919.sdrf.txt')

    gtex_annot_input_path_2 = os.path.join('/opt/ml/processing/input', 'E-MTAB-5214.sdrf.txt')

    
    tcga_df = pd.read_csv(tcga_input_path, sep="\t", index_col=0)
    print('Reading input data from {}'.format(tcga_input_path))

    tcga_df = tcga_df.T
    tcga_df.columns = [tcga_df.columns[i] + "_" + str(i) for i in range(len(tcga_df.columns))]

    tcga_df = tcga_df.astype(np.float16)

    print('Reading input data from {}'.format(tcga_annot_input_path))
    df = pd.read_csv(tcga_annot_input_path, sep="\t", index_col=0)

    print('Start processing TCGA Data')

    df["abbreviated_disease"] = df["_primary_disease"].apply(lambda x: TCGA_MAP[x])

    str_cols = ["sample_type", "_primary_disease", "abbreviated_disease"]

    for col in str_cols:

        df[col] = df[col].astype(str)

    df.index = df.index.astype(str)
    df.abbreviated_disease = pd.Categorical(df.abbreviated_disease)
    annot_df = df
    
    print('Join TCGA RNA and Annotation')

    tdf = tcga_df.join(annot_df, how="inner")
    tcga_y = 1+tdf.abbreviated_disease.cat.codes.values
    tcga_diseases = tdf.abbreviated_disease
    tdf = tdf.iloc[:,:-annot_df.shape[1]]
    
    import pandas as pd
    from importlib import reload
    reload(pd)

    print('Reading input data from {}'.format(gtex_input_path))

    gtex_df = pd.read_csv(gtex_input_path, skiprows=2, index_col=0, sep="\t")
    gtex_df.index = gtex_df["Description"].str.cat(["_"+v for v in gtex_df.index.values])
    gtex_df.drop(["Description"], axis=1, inplace=True)
    gtex_df = gtex_df.T

    gtex_df = np.log2(gtex_df + 1)
    gtex_df = gtex_df.astype(np.float16)
    
    print('Reading input data from {}'.format(gtex_annot_input_path_1))

    gtex_annot_df = pd.concat([pd.read_csv(gtex_annot_input_path_1, sep="\t"), pd.read_csv(gtex_annot_input_path_2, sep="\t")], axis=0, sort=True)
    
    print('Starting to process GTExdata')

    gtex_annot_df = gtex_annot_df.set_index("Source Name")
    gtex_annot_df = gtex_annot_df[~gtex_annot_df.index.duplicated(keep='first')]
    gdf = gtex_df.join(gtex_annot_df, how="inner")
    gdf = gdf.iloc[:,:gtex_df.shape[1]]
    
    print('Finding Common Genes')
    gtextgenes = [x.split("_")[0] for x in gdf.columns]
    tcgagenes = [x.split("_")[0] for x in tdf.columns]
    common_genes = np.intersect1d(gtextgenes, tcgagenes).tolist()

    # get rid of unused suffixes to harmonize column names
    t,g = tdf.rename(columns=lambda x: x.split("_")[0])[common_genes], gdf.rename(columns=lambda x: x.split("_")[0])[common_genes]

    # get rid of duplicate columns
    
    # get rid of duplicate columns
    def drop_duplicate_columns(d):
        uq, cnt = np.unique(d.columns,return_counts=True)
        duplicates = np.where(cnt>1)[0]
        return d.drop(uq[duplicates],axis=1)
    t,g = drop_duplicate_columns(t), drop_duplicate_columns(g)
    
    # find intersection (common columns)
    common_genes = np.intersect1d(t.columns, g.columns).tolist()
    print(t[common_genes].shape,g[common_genes].shape)

    # concatenate
    df = pd.concat((t[common_genes], g[common_genes]), axis=0)
    print(df.shape)

    print('Concatenate data sets')

    y = np.concatenate((tcga_y, [0]*len(gdf)))
    
    from scipy.stats import itemfreq
    F =  itemfreq(tcga_diseases)
    print(F)
    
    F = F[np.argsort(F[:,-1])[::-1]]
    print(F)
    
    classes, cnt = np.unique(y, return_counts=True)
    most_freq_count_idx = np.argsort(cnt)[::-1][:6]
    most_freq_cls = list(classes[most_freq_count_idx])
    idx5 = [i for i in range(len(y)) if y[i] in most_freq_cls and y[i] != 0]
    y5 = np.array([most_freq_cls.index(c) for c in y[idx5]])

    print('Size of y5: '+str(y5.size))

    all_model_df = df.iloc[idx5]
    print('Size of x: '+str(all_model_df.shape))
    
    #all_model_df.reset_index().to_feather("all_model_df.feather")
    nans = all_model_df.isna()
    drop_features = []

    for c in np.unique(y5):
        feature_wise_nan_prevalence_c = nans[y5==c].mean(0)
        feature_wise_nan_prevalence_nonc = nans[y5!=c].mean(0)
        nan_is_discriminative = np.where(feature_wise_nan_prevalence_nonc != feature_wise_nan_prevalence_c)[0]
        drop_features += list(nan_is_discriminative)
        print(c,len(nan_is_discriminative))
    
    print(all_model_df.shape)
    all_model_df.drop(all_model_df.columns[np.unique(drop_features)],axis=1,inplace=True)
    print(all_model_df.shape)
    
    y_features_output_path = os.path.join('/opt/ml/processing/train', 'y5.npy')
    all_model_df_output_path = os.path.join('/opt/ml/processing/train', 'all_model_df.csv')
    
    print('Saving y features to {}'.format(y_features_output_path))
    np.save(y_features_output_path,y5)

    #pd.DataFrame(y5).to_csv(y_features_output_path, header=False, index=False)
    
    print('Saving all model df features to {}'.format(all_model_df_output_path))
    pd.DataFrame(all_model_df).to_csv(all_model_df_output_path, header=True, index=False)
