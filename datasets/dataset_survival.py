#输入数据要求 
#.pt或.h5格式特征文件
#组学特征  每一行为不同的case_id    每一列对应RNA-seq、CNV、Mutation之类的基因表达矩阵
#生存信息   OS_time：生存时长  censorship：是否发生事件（如0=未删失，1=删失
#可把所有信息整合进一个大的csv文件！！推荐



from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self,
        csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False,
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None
        #下面的if顺序错了，应该把读取slidedata的放最上面
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)

        slide_data = pd.read_csv(csv_path, low_memory=False)
        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        #提取slide_data文件夹里的index的前12个字符作为case_id列    
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        #定义结果标签
        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col
        #如果处理的为乳腺癌，那么只用IDC类型的肿瘤样本，下面两行只为BRCA数据集的筛选逻辑
        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']
        #筛选出非重复且未删失的病例（这里认为0为未删失），删失的数据同样重要，有些方法支持使用删失数据
        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]
        #对生存时间进行分箱处理
        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        #把所有病例按照上面的分箱结果进行分箱
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))
        #构建patient_dict，里面包含每一个case_id对应的slide_id，如果一个病人有多个slide，则都对应起来
        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})
        
        self.patient_dict = patient_dict
        #把去重后的patients_df重命名为slide_data
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])
        #生存分箱编号(分箱编号 i, 删除标志 c)：接0，1，2，3，4，5······，总的来说就是根据生存分箱和删失与否自动生成分类标签
        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1
        #将前面生成的label_dict应用到每个样本上；结果就是disc_label列为生存时间分箱，label列为生存时间与删失状态组合起来的分箱
        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]
        #实例化生存分箱，检查总分箱数量，确保patients_df的每一行都是单独病人不重复，实例化一个两列的表，一列是case_id，一列是总分箱
        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}
        #把最后一列移动到最前面，实例化slide_data，定义一个普通变量为metadata，实例化slide_data的前12列为self.metadata
        #new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2]) ### ICCV
        new_cols = list(slide_data.columns[-1:]) + list(slide_data.columns[:-1])  ### PORPOISE
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        metadata = ['disc_label', 'Unnamed: 0', 'case_id', 'label', 'slide_id', 'age', 'site', 'survival_months', 'censorship', 'is_female', 'oncotree_code', 'train']
        self.metadata = slide_data.columns[:12]
        #循环slide_data中所有的非self.metadata列，判断其是否符合预设的命名规则，不符合的将被打印出来
        for col in slide_data.drop(self.metadata, axis=1).columns:
            if not pd.Series(col).str.contains('|_cnv|_rnaseq|_rna|_mut')[0]:
                print(col)
        #pdb.set_trace()
        #确保self.metadata与metadata的顺序内容等是一样的，把运行脚本传入的mode参数实例化为self.mode
        assert self.metadata.equals(pd.Index(metadata))
        self.mode = mode
        self.cls_ids_prep()

        ### ICCV discrepancies
        # For BLCA, TPTEP1_rnaseq was accidentally appended to the metadata
        #pdb.set_trace()
        #打印数据摘要信息
        if print_info:
            self.summarize()
        #如果运行脚本传入了apply_sig store_true参数，那么就实例化./datasets_csv_sig/signatures.csv文件
        #注意如果想使用apply_sig store_true参数要创建一个对应的./datasets_csv_sig/signatures.csv目录  
        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()

    #为不同的分箱结果创建对应的索引，比如分箱为1的病人序号都为哪些
    def cls_ids_prep(self):
        r"""

        """
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

    #提取所有唯一病人ID
    def patient_data_prep(self):
        r"""
        
        """
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        #遍历所有病人，保存病人ID与总分箱结果便于后续使用
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}

    #处理标签列的预处理函数
    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        r"""
        
        """

        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins
    #定义长度函数，根据需求返回病人或slides的数量
    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)
    #定义一个名为 summarize 的类方法，作用是打印出当前数据集的关键信息和统计数据
    def summarize(self):
        print("label column: {}".format(self.label_col))#输出当前使用的标签列的名字
        print("label dictionary: {}".format(self.label_dict))#打印标签字典，即将标签值映射为内部编号的映射表
        print("number of classes: {}".format(self.num_classes))#输出类别的总数，也就是任务中有多少个不同的分类标签
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))#统计并打印所有切片（slide）级别的标签分布情况
        for i in range(self.num_classes):#这个循环依次打印每个类别在病人级别和切片级别的样本数量
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))

    #定义一个函数，使其能从一个包含所有划分信息的字典 all_splits 中取出指定子集（如 'train'、'val'、'test'）
    #并在当前数据中筛选出对应的 slide，构造新的数据子集对象
    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]#从 all_splits 中取出对应 split_key 的那一部分切片 ID 列表或序列
        split = split.dropna().reset_index(drop=True)#移除na值，并重建移除na值之后的表的索引

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, signatures=self.signatures, data_dir=self.data_dir, label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split

    #调用刚刚定义的函数提取对应的数据集
    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            #归一化前面得到的子集
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split#, test_split

    #定义根据索引返回对应信息的函数
    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None

#定义一个复杂类函数，并且使其运行时传入的参数可以同步传入到上面定义的Generic_WSI_Survival_Dataset中
class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False  #默认不使用.h5格式的文件
    #定义一个函数，把toggle 赋值给 self.use_h5判断是否用.h5文件
    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = torch.Tensor([self.slide_data['disc_label'][idx]])
        event_time = torch.Tensor([self.slide_data[self.label_col][idx]])
        c = torch.Tensor([self.slide_data['censorship'][idx]])
        slide_ids = self.patient_dict[case_id]
        #判断特征数据目录self.data_dir为字符串还是字典，并对应的加载
        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        #判断使用.h5还是.pt（这里.pt文件进入下一步）
        if not self.use_h5:
            if self.data_dir:
                 #按病人加载多个对应病人的 slide 特征并拼接返回
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label, event_time, c)
                #相较于 'path' 模式多处理了聚类 ID和基因组特征
                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)
                #只使用 基因组学特征（omics data） 来进行建模
                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features.unsqueeze(dim=0), label, event_time, c)
                #病理图片特征+基因组学联合建模
                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), label, event_time, c)
                #直接加载已经拼接好的病人及的病理图片特征，来加速
                elif self.mode == 'pathomic_fast':
                    casefeat_path = os.path.join(data_dir, f'split_{self.split_id}_case_pt', f'{case_id}.pt')
                    path_features = torch.load(casefeat_path)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features.unsqueeze(dim=0), label, event_time, c)
                #使用 co-attention（协同注意力）机制 来联合处理病理图像特征和多组组学（omics）特征时数据的加载
                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path)
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx])
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx])
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx])
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx])
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx])
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx])
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)

                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, 
        signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures

        if mode == 'cluster':
            with open(os.path.join(data_dir, 'fast_cluster_ids.pkl'), 'rb') as handle:
                self.fname2ids = pickle.load(handle)

        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))

        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--

    def set_split_id(self, split_id):
        self.split_id = split_id
