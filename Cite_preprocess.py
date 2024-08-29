import os
import pandas as pd
import numpy as np
import tables
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

DATA_DIR = "D:/kaggle/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")
FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")
FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")
"""
FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")
"""
#FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
#FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

save_dir ="D:/kaggle/processed/"
os.makedirs(save_dir, exist_ok=True)

final_cite_train_hdf5_path = os.path.join(save_dir, "processed_cite_train_data.h5")
final_cite_test_hdf5_path = os.path.join(save_dir, "processed_cite_test_data.h5")
final_cite_target_hdf5_path = os.path.join(save_dir, "processed_cite_target_data.h5")

#df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
# Read the data
df_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS,start=0,stop=200)
df_cite_test_x = pd.read_hdf(FP_CITE_TEST_INPUTS,start=0,stop=200)
df_cite_target_data=pd.read_hdf(FP_CITE_TRAIN_TARGETS,start=0,stop=200)


def preprocessing1(data):
    #1. Remove genes where all cells have 0 expression
    genes_to_remove = data.loc[:, (data != 0).sum(axis=0) == 0].columns
    processed_data = data.drop(columns=genes_to_remove)

    # 2. Divide count data by the row-wise non-zero median values
    nonzero_medians = processed_data.apply(lambda x: np.median(x[x > 0]), axis=1)
    processed_data = processed_data.divide(nonzero_medians, axis=0)

    # 3. Transform with log1p
    processed_data= processed_data.apply(np.log1p)
    return processed_data

def tsvd_impute(data, n_components):
    # 记录原始数据中的零值位置
    zero_mask = (data == 0)

    # Step 1: 对数据进行 tSVD 分解并进行插补
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    imputed_data = svd.fit_transform(data)
    imputed_data = svd.inverse_transform(imputed_data)
    if imputed_data.shape != data.shape:
        print(imputed_data.shape)
        print(data.shape)
    # 将原始数据中的零值用插补后的值替换
    imputed_data = pd.DataFrame(imputed_data, index=data.index, columns=data.columns)
    data[zero_mask] = imputed_data[zero_mask]
    
    return data

def subtract_column_median(data):
    # Step 2: 从每列减去中位数
    column_medians = data.median(axis=0)
    data_subtracted = data - column_medians
    return data_subtracted

def perform_tsvd(data, n_components):
    # Step 3: 使用 tSVD 进行降维
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_data = svd.fit_transform(data)
    return reduced_data

# 假设 df_cite_train_x 和 df_cite_test_x 已经被读取并经过预处理


def SVDpreprocessing(data,n_components_for_imputation,n_components_for_reduction):
    original_index=data.index 
    reduced_data=tsvd_impute(data,n_components_for_imputation)
    reduced_data=subtract_column_median(reduced_data)
    reduced_data=perform_tsvd(reduced_data,n_components_for_reduction)
    reduced_data_df = pd.DataFrame(reduced_data, index=original_index)
    return reduced_data_df

def save_data_with_metadata(file_path, data_frame):
    with tables.File(file_path, mode='w') as hdf_file:
        # 保存数据
        hdf_file.create_carray(
            where='/',
            name='df',
            obj=data_frame.to_numpy(),
            chunkshape=(100, data_frame.shape[1]),  # 可以根据需要调整块大小
            filters=tables.Filters(complevel=9)
        )
        # 保存列名作为独立的数组
        column_names = data_frame.columns.tolist()
        
        hdf_file.create_array(
            where='/',
            name='columns',
            obj=np.array(column_names, dtype='S')
            
        )
        # 保存行标签作为独立的数组
        row_labels = data_frame.index.tolist()
        hdf_file.create_array(
            where='/',
            name='row_labels',
            obj=np.array(row_labels, dtype='S')
        )

def input_preprocessing():
    
    n_components_for_imputation = 128  
    n_components_for_reduction = 128  

    all_data=pd.concat([df_cite_train_x,df_cite_train_x])
    
    processed_aLL_data=preprocessing1(all_data)

    reduced_data=SVDpreprocessing(processed_aLL_data,n_components_for_imputation,n_components_for_reduction)

    final_all_data=pd.concat([processed_aLL_data, reduced_data], axis=1)

    final_cite_train_data = final_all_data.iloc[:len(df_cite_train_x)]
    final_cite_test_data = final_all_data.iloc[len(df_cite_train_x):]

    """
    processed_cite_train_x=preprocessing1(df_cite_train_x)
    processed_cite_test_x=preprocessing1(df_cite_train_x)
    #print(processed_cite_train_x)

    reduced_cite_train_data=SVDpreprocessing(processed_cite_train_x,n_components_for_imputation,n_components_for_reduction)
    reduced_cite_test_data=SVDpreprocessing(processed_cite_test_x,n_components_for_imputation,n_components_for_reduction)

    #print(reduced_cite_test_data.shape)
    #print(reduced_cite_train_data.shape)

    final_cite_train_data = pd.concat([processed_cite_train_x, reduced_cite_train_data], axis=1)
    final_cite_test_data = pd.concat([processed_cite_test_x, reduced_cite_test_data], axis=1)

    #final_cite_train_data.index.name = 'cell_id'
    #final_cite_test_data.index.name = 'cell_id'
    #print(final_cite_test_data.columns)
    #上述曾用名为combined

    """

    """
    # 将metadata中的cell_type与train数据合并
    final_cite_train_data = combined_cite_train_data.merge(df_meta[['cell_type', 'donor', 'day']], left_index=True, right_index=True, how='left')

    # 将metadata中的cell_type与test数据合并
    final_cite_test_data = combined_cite_test_data.merge(df_meta[['cell_type', 'donor', 'day']], left_index=True, right_index=True, how='left')

    # 输出合并后的数据

    print(final_cite_train_data)
    print(final_cite_test_data)
    print(final_cite_train_data.shape)
    print(final_cite_test_data.shape)
    因为meta中的celltype不能存进HDF5 所以不在这里进行合并了
    """
    

    #print(final_cite_train_data.dtypes)
    #print(final_cite_test_data.dtypes)
    """
    # 将'cell type' 'donor' 和 'day' 列转化为 category 类型
    final_cite_train_data['cell_type'] = final_cite_train_data['cell_type'].astype(str)
    final_cite_train_data['donor'] = final_cite_train_data['donor'].astype(str)
    final_cite_train_data['day'] = final_cite_train_data['day'].astype(str)


    final_cite_test_data['cell_type'] = final_cite_test_data['cell_type'].astype(str)
    final_cite_test_data['donor'] = final_cite_test_data['donor'].astype(str)
    final_cite_test_data['day'] = final_cite_test_data['day'].astype(str)

    #print(final_cite_train_data)
    #print(final_cite_test_data)
    print(final_cite_train_data.dtypes)
    print(final_cite_test_data.dtypes)
    """
    #final_cite_train_hdf5_path = "processed_cite_train_data.h5"
    #final_cite_train_data.to_hdf(final_cite_train_hdf5_path, key='df', mode='w', format='table',complevel=9, complib='blosc', chunksize=100) 

    #final_cite_test_hdf5_path = "processed_cite_test_data.h5"
    #final_cite_train_data.to_hdf(final_cite_test_hdf5_path, key='df', mode='w', format='table',complevel=9, complib='blosc', chunksize=100) 

    #final_cite_train_data_array = final_cite_train_data.to_numpy()
    save_data_with_metadata(final_cite_train_hdf5_path, final_cite_train_data)

    #final_cite_test_data_array = final_cite_test_data.to_numpy()
    save_data_with_metadata(final_cite_test_hdf5_path, final_cite_test_data)

#------------------------------input data preprocessing finished--------------------------------

def target_preprocessing():
    

    original_row_labels=df_cite_target_data.index

    # 步骤 1: 对每一行进行标准化
    scaler = StandardScaler()
    normalized_target_data = scaler.fit_transform(df_cite_target_data)

    # 步骤 2: 每列数据减去对应的中位数
    column_medians = np.median(normalized_target_data, axis=0)
    median_subtracted_data = normalized_target_data - column_medians

    # 步骤 3: 使用 TruncatedSVD 将数据降至128维
    tsvd = TruncatedSVD(n_components=128)
    reduced_data = tsvd.fit_transform(median_subtracted_data)


    column_names = [f'component_{i}' for i in range(128)]  # 生成从 0 到 127 的列名
    reduced_data = pd.DataFrame(reduced_data, index=original_row_labels, columns=column_names)
    print(reduced_data)


    # 保存降维后的数据和相关信息
    with tables.File(final_cite_target_hdf5_path, mode='w') as hdf_file:
        # 保存降维数据
        hdf_file.create_carray(
            where='/',
            name='reduced_data',
            obj=reduced_data.to_numpy(),
            chunkshape=(100, reduced_data.shape[1]),  # 可以根据需要调整块大小
            filters=tables.Filters(complevel=9)
        )
        
        # 保存列名
        column_names = reduced_data.columns.tolist()
        hdf_file.create_array(
            where='/',
            name='original_columns',
            obj=np.array(column_names, dtype='S')
        )
        
        # 保存行标签
        row_labels = reduced_data.index.tolist()
        hdf_file.create_array(
            where='/',
            name='original_row_labels',
            obj=np.array(row_labels, dtype='S')
        )

        # 保存逆变换所需的参数
        hdf_file.create_array(
            where='/',
            name='scaler_mean',
            obj=scaler.mean_
        )
        
        hdf_file.create_array(
            where='/',
            name='scaler_scale',
            obj=scaler.scale_
        )
        
        hdf_file.create_array(
            where='/',
            name='column_medians',
            obj=column_medians
        )
        
        hdf_file.create_carray(
            where='/',
            name='tsvd_components',
            obj=tsvd.components_,
            chunkshape=(100, tsvd.components_.shape[1]),
            filters=tables.Filters(complevel=9)
        )
        
        hdf_file.create_array(
            where='/',
            name='tsvd_explained_variance',
            obj=tsvd.explained_variance_
        )
        
        hdf_file.create_array(
            where='/',
            name='tsvd_explained_variance_ratio',
            obj=tsvd.explained_variance_ratio_
        )




print(df_cite_target_data)
print(df_cite_target_data.columns)

target_preprocessing()
input_preprocessing()