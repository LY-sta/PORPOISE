from __future__ import print_function
import numpy as np
import torch_geometric

import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler


def main(args):
	#### Create Results Directory
	# 创建结果目录
	if not os.path.isdir(args.results_dir):
		os.mkdir(args.results_dir)
        #确定交叉验证起止轮次
	if args.k_start == -1:
		start = 0
	else:
		start = args.k_start
	if args.k_end == -1:
		end = args.k
	else:
		end = args.k_end
        #准备记录变量
	latest_val_cindex = []
	folds = np.arange(start, end)

	### Start 5-Fold CV Evaluation.
	#5-Fold CV主循环
	for i in folds:
		#记录开始时间,方便后续计算每fold耗时
		start = timer()
		#设置随机种子
		seed_torch(args.seed)
		#检查当前结果是否已存在，是否覆盖,如果当前fold结果已经算过，且没有要求强制重算，就直接跳过，节省时间
		results_pkl_path = os.path.join(args.results_dir, 'split_latest_val_{}_results.pkl'.format(i))
		if os.path.isfile(results_pkl_path) and (not args.overwrite):
			print("Skipping Split %d" % i)
			continue

		### Gets the Train + Val Dataset Loader.
		#加载训练/验证集
		train_dataset, val_dataset = dataset.return_splits(from_id=False, 
				csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
		train_dataset.set_split_id(split_id=i)
		val_dataset.set_split_id(split_id=i)
		
		#pdb.set_trace()
		print('training: {}, validation: {}'.format(len(train_dataset), len(val_dataset)))
		datasets = (train_dataset, val_dataset)
		
		### Specify the input dimension size if using genomic features.
		#设置输入特征维数
		if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
			args.omic_input_dim = train_dataset.genomic_features.shape[1]
			print("Genomic Dimension", args.omic_input_dim)
		elif 'coattn' in args.mode:
			args.omic_sizes = train_dataset.omic_sizes
			print('Genomic Dimensions', args.omic_sizes)
		else:
			args.omic_input_dim = 0

		### Run Train-Val on Survival Task.
		#调用主训练函数，执行训练+验证
		if args.task_type == 'survival':
			val_latest, cindex_latest = train(datasets, i, args)
			latest_val_cindex.append(cindex_latest)

		### Write Results for Each Split to PKL
		#保存当前fold的结果到pkl文件
		save_pkl(results_pkl_path, val_latest)
		end = timer()
		print('Fold %d Time: %f seconds' % (i, end - start))

	### Finish 5-Fold CV Evaluation.
	#5折交叉验证结束，整合所有结果
	if args.task_type == 'survival':
		results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

	if len(folds) != args.k:
		save_name = 'summary_partial_{}_{}.csv'.format(start, end)
	else:
		save_name = 'summary.csv'

	results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))

### Training settings
#使用该脚本的时候，直接在运行脚本的时候python main.py --data_root_dir /data/tcga这样子就能直接指定数据存放目录
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
#病理图片特征部分
#原始数据（如组学、病理图片等WSI特征）的存放目录
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir', help='Data directory to WSI features (extracted via CLAM')
#随机种子，保证每次实验结果一致，可复现
parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
#k折交叉验证的折数，默认5
parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
#只训练部分fold时指定起始和终止的fold号
parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
#输出结果文件保存路径
parser.add_argument('--results_dir',     type=str, default='./results_new', help='Results directory (Default: ./results)')
#指定用哪种分割方案（如5折/10折/自定义分组），其实就是用哪个split目录
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
#哪种癌症/任务名/子目录名，进一步细分split文件夹
parser.add_argument('--split_dir',       type=str, default='tcga_blca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
#是否用tensorboard记录训练日志
parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
#如果已经有结果文件，是否覆盖旧实验
parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

### Model Parameters.
#模型相关的参数设置
#控制主流程要用哪种核心模型
parser.add_argument('--model_type',      type=str, default='mcat', help='Type of model (Default: mcat)')
#决定本次训练用哪些数据源、怎么拼接/融合数据
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'pathomic_fast', 'cluster', 'coattn'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
#控制不同模态的特征是如何合并的
parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='None', help='Type of fusion. (Default: concat).')
#通常指用特定的生物marker gene的表达向量作为输入，而不是全量组学特征
parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
#是否把组学特征直接当作表格变量
parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
#控制是否在网络中加dropout（p=0.25），防止过拟合
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
#控制WSI路径特征处理网络的宽度/深度（如small/big）
parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
#控制组学特征分支的全连接层规模（如small/big）
parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

#输出类别数
parser.add_argument('--n_classes', type=int, default=4)


### PORPOISE
#控制是否应用mutsig，mutsig通常用于突变签名分析
parser.add_argument('--apply_mutsig', action='store_true', default=False)
#控制是否在模型中使用病理图片（Pathology）相关的gate机制
parser.add_argument('--gate_path', action='store_true', default=False)
#控制是否在模型中使用组学数据（Omics）相关的gate机制
parser.add_argument('--gate_omic', action='store_true', default=False)
#控制模型中特定维度的缩放比例，用于调整输入或中间层的维度
parser.add_argument('--scale_dim1', type=int, default=8)
parser.add_argument('--scale_dim2', type=int, default=8)
#控制是否跳过某些操作
parser.add_argument('--skip', action='store_true', default=False)
#控制输入层的dropout比例。dropout是深度学习中常用的防止过拟合的技术，通过随机丢弃神经网络中的部分节点来提高模型的泛化能力
parser.add_argument('--dropinput', type=float, default=0.0)
#指定病理图片输入特征的维度,这个参数是用来控制输入病理特征（如WSI特征提取后的向量）的维度大小,根据实际使用的病理图片特征的维度，调整这个参数
parser.add_argument('--path_input_dim', type=int, default=1024)
#控制是否使用**MLP（多层感知机）**作为网络的一部分
parser.add_argument('--use_mlp', action='store_true', default=False)


### Optimizer Parameters + Survival Loss Function
#优化器参数和损失函数相关的命令行参数
#可选adam（自适应学习率，收敛快，推荐用于大部分深度模型）、sgd（传统随机梯度下降，适合大数据和线性模型）
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
#每次训练的样本批量大小,多实例学习（MIL）等任务由于每个样本“袋子”里的实例数量不同，经常用batch_size=1保证训练的灵活性（每次喂一个病人/切片/包）
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
#梯度累积步数，用于“模拟大batch训练”，即实际每处理gc个batch再做一次梯度更新，适合显存小但想用大batch的情况
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
#最大训练轮数，到达指定epoch后自动终止训练
parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
#学习率，控制每次参数更新的步长，是模型能否收敛、收敛快慢的关键参数。
parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
#包级（slide-level）损失函数类型，可选：svm: 支持向量机损失,ce: 交叉熵（多分类常用）,ce_surv: 生存分析任务用的交叉熵变体,nll_surv: 生存分析的负对数似然损失（更适合处理删失数据）
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
#训练集标签采样比例，用于半监督/弱监督/标签不全实验，1.0代表全部标签都用，0.5就是只随机用一半标签。
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
#L2正则化（权重衰减），用于防止过拟合，越大模型越保守。避免了模型对特定数据的过度依赖（即过拟合）
parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
#对“未删失患者”的损失加权系数（生存分析常用），调整模型对不同类型患者的关注度。
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
#L1正则化的作用对象：'omic'：仅对组学分支加L1正则，'pathomic'：对融合层或多模态层加L1，'None'：不加
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
#L1正则化系数，越大特征稀疏性越强，可用于特征选择。
parser.add_argument('--lambda_reg',      type=float, default=1e-5, help='L1-Regularization Strength (Default 1e-4)')
#启用带权重的采样，用于处理数据类别不平衡（如正负样本极不均衡时，按类别权重采样）。
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
#启用Early Stopping，在验证集loss长时间不下降时提前停止训练，防止过拟合。
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

### CLAM-Specific Parameters
#bag_weight=0.7**意味着模型将70%的关注放在袋子级别的预测上
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
#调试工具，如果设置了--testing，则args.testing会被设置为True，启用--testing模式通常意味着开启一些简化的训练设置如使用更小的数据集进行测试，简化模型结构，只运行部分训练周期
parser.add_argument('--testing', 	 	 action='store_true', default=False, help='debugging tool')

#解析命令行传入的所有参数，将它们存储在args对象中，之后可以通过args.xxx来访问这些参数
args = parser.parse_args()
#如果有可用的CUDA设备（即GPU），则将设备设置为GPU（cuda）,否则则回退到CPU（cpu）
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
#根据传入的参数（比如split_dir、task等），将它们组合起来，生成一个标准的、具有描述性的实验代码，方便后续的实验管理和结果保存。
args = get_custom_exp_code(args)
#生存分析任务（survival），任务名称由split_dir（分割文件夹名）推断而来
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
#打印出生成的实验名称
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
	import random
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if device.type == 'cuda':
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k,     # 交叉验证的折数
			'k_start': args.k_start,                  # 起始的折数（通常用于选择部分折）
			'k_end': args.k_end,                      # 结束的折数
			'task': args.task,                        # 任务类型（如生存分析、分类等）
			'max_epochs': args.max_epochs,            # 最大训练轮数
			'results_dir': args.results_dir,          # 结果保存目录
			'lr': args.lr,                            # 学习率
			'experiment': args.exp_code,              # 实验代码，用于唯一标识当前实验
			'reg': args.reg,                          # L2正则化的权重衰减
			'label_frac': args.label_frac,            # 训练集标签使用的比例
			'bag_loss': args.bag_loss,                # Bag-level损失函数（如SVM、交叉熵等）
			#'bag_weight': args.bag_weight,           # Bag-level损失权重（目前注释掉了）
			'seed': args.seed,                        # 随机种子
			'model_type': args.model_type,            # 模型类型（如SNN、AMIL等）
			'model_size_wsi': args.model_size_wsi,    # WSI（病理图像）模型的大小
			'model_size_omic': args.model_size_omic,  # 组学数据模型的大小
			"use_drop_out": args.drop_out,            # 是否启用dropout
			'weighted_sample': args.weighted_sample,  # 是否启用加权采样
			'gc': args.gc,                            # 梯度累积步数
			'opt': args.opt}                          # 优化器类型（如Adam、SGD等）
print('\nLoad Dataset')
#：根据args.task判断当前任务是否为生存分析任务。如果是生存分析任务，则加载相关数据集并进行配置
if 'survival' in args.task:
	study = '_'.join(args.task.split('_')[:2])
	if study == 'tcga_kirc' or study == 'tcga_kirp':
		combined_study = 'tcga_kidney'
	elif study == 'tcga_luad' or study == 'tcga_lusc':
		combined_study = 'tcga_lung'
	else:
		combined_study = study
	
	study_dir = '%s_20x_features' % combined_study
	#数据集加载：使用Generic_MIL_Survival_Dataset加载数据集，并根据配置的参数处理数据
	dataset = Generic_MIL_Survival_Dataset(csv_path = './%s/%s_all_clean.csv.zip' % (args.dataset_path, study),
										   mode = args.mode,
										   apply_sig = args.apply_sig,
										   data_dir= os.path.join(args.data_root_dir, study_dir),
										   shuffle = False, 
										   seed = args.seed, 
										   print_info = True,
										   patient_strat= False,
										   n_bins=4,
										   label_col = 'survival_months',
										   ignore=[])
else:
	raise NotImplementedError
#检查数据集是否是Generic_MIL_Survival_Dataset类型。如果是，它会将任务类型args.task_type设置为'survival'，表示当前任务是生存分析任务
if isinstance(dataset, Generic_MIL_Survival_Dataset):
	args.task_type = 'survival'
else:
	raise NotImplementedError

### Creates results_dir Directory.
#创建存储实验结果的目录。它会检查指定的结果文件夹是否存在，如果不存在，则会创建该目录
if not os.path.isdir(args.results_dir):
	os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
#根据指定的路径，动态创建一个更具体的实验结果文件夹，并检查该文件夹是否已经存在以避免覆盖已有的结果
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed))
if not os.path.isdir(args.results_dir):
	os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
	print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
	sys.exit()

### Sets the absolute path of split_dir
#设置了args.split_dir的绝对路径，并将其与当前目录中的splits文件夹结合起来
#通过os.path.join()，路径会被动态拼接成像这样的完整路径：'./splits/5foldcv/tcga_kirc'
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
#os.path.isdir(args.split_dir)会检查args.split_dir是否存在且是一个目录
assert os.path.isdir(args.split_dir)
#将args.split_dir添加到settings字典中，确保实验设置中包含数据划分的目录路径
settings.update({'split_dir': args.split_dir})

#将当前实验的设置保存到一个文本文件中
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
	print(settings, file=f)
f.close()

#打印出所有的实验设置，这部分代码会遍历settings字典，并逐项打印出每个配置的键（key）和值（val），方便查看当前实验的具体设置
print("################# Settings ###################")
for key, val in settings.items():
	print("{}:  {}".format(key, val))        

#确保当该脚本作为主程序运行时，以下代码才会执行
if __name__ == "__main__":
	#记录程序开始的时间
	start = timer()
	#调用main函数开始运行实验，main(args)会接收命令行参数并开始执行实际的训练/评估任务
	results = main(args)
	#记录程序结束的时间
	end = timer()
	print("finished!")
	print("end script")
	#打印整个脚本的运行时间
	print('Script Time: %f seconds' % (end - start))
