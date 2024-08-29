import os
import pandas as pd
import numpy as np
import tables
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/home/jdhan_pkuhpc/profiles/lijianzhe/gpfs1/open-problems-multimodal/"
FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")
FP_MULTIOME_TEST_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")
FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

#FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")
#FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

save_dir ="/home/jdhan_pkuhpc/profiles/lijianzhe/gpfs1/processed/"
os.makedirs(save_dir, exist_ok=True)

final_multi_train_hdf5_path = os.path.join(save_dir, "processed_multi_train_data.h5")
final_multi_test_hdf5_path = os.path.join(save_dir, "processed_multi_test_data.h5")
final_multi_target_hdf5_path = os.path.join(save_dir, "processed_multi_target_data.h5")
concat_train_data_path=os.path.join(save_dir, "concat_multi_input_data.h5")
combined_data_reduced_1_path=os.path.join(save_dir, "combined_data_reduced_1.h5")
#df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
# Read the data




def preprocessing_remove(data):
    #1. Remove genes where all cells have 0 expression
    genes_to_remove = data.loc[:, (data != 0).sum(axis=0) == 0].columns
    processed_data = data.drop(columns=genes_to_remove)

   
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

def perform_tsvd(data, n_components=128):
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

def normalize_by_nonzero_median(df):
    median_vals = df[df != 0].median(axis=1)
    df_normalized = df.div(median_vals, axis=0)
    return df_normalized.fillna(0)

def input_preprocessing():
    df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS)
    df_multi_test_x = pd.read_hdf(FP_MULTIOME_TEST_INPUTS)

    print("train_data:")
    print()
    print(df_multi_train_x.shape)
    print(df_multi_train_x)

    print("test_data:")
    print()
    print(df_multi_test_x.shape)
    print(df_multi_test_x)

    combined_data = pd.concat([df_multi_train_x, df_multi_test_x], axis=0)
    combined_data=preprocessing_remove(combined_data)
    
    combined_data = normalize_by_nonzero_median(combined_data)
    save_data_with_metadata(concat_train_data_path,combined_data)
    #combined_data = pd.concat([df_multi_train_x_normalized, df_multi_test_x_normalized], axis=0)

# 进行tSVD降维
    combined_data_reduced_1 = perform_tsvd(combined_data)
    print(combined_data_reduced_1)
    
    save_data_with_metadata(combined_data_reduced_1_path,combined_data_reduced_1)
# 将非零值转换为 1
    """
    combined_data = pd.concat([df_multi_train_x, df_multi_test_x], axis=0)
    combined_data=preprocessing_remove(combined_data)

    combined_data = combined_data.applymap(lambda x: 1 if x != 0 else 0)

    combined_data_reduced_2=perform_tsvd(combined_data)
    del combined_data
    print(combined_data_reduced_2)
    combined_data_reduced=pd.concat([combined_data_reduced_1,combined_data_reduced_2],axis=1)
    
# 拆分降维后的数据

    df_multi_train_reduced = combined_data_reduced.iloc[:len(df_multi_train_x)]
    df_multi_test_reduced = combined_data_reduced.iloc[len(df_multi_train_x):]

    save_data_with_metadata(final_multi_train_hdf5_path, df_multi_train_reduced)
    save_data_with_metadata(final_multi_test_hdf5_path, df_multi_test_reduced)


    print()
    print("final_multi_train_data:")
    print(df_multi_train_reduced.shape)
    print(df_multi_train_reduced)
    print()
    print("final_multi_test_data:")
    print(df_multi_test_reduced.shape)
    print(df_multi_test_reduced)
    """
#------------------------------input data preprocessing finished--------------------------------

def target_preprocessing():
    df_multi_target_data = pd.read_hdf(FP_MULTIOME_TEST_TARGETS)

    print("target_data:")
    print()
    print(df_multi_target_data)
    print(df_multi_target_data.shape)

    row_medians = np.median(df_multi_target_data[df_multi_target_data != 0], axis=1)
    row_medians[row_medians == 0] = 1  # 防止除以零
    median_divided_data = df_multi_target_data.div(row_medians, axis=0)

# 步骤 2: 进行log1p变换
    log_transformed_data = np.log1p(median_divided_data)

# 步骤 3: 用tSVD方法对0值进行填充
    tsvd_imputer = TruncatedSVD(n_components=128)
    tsvd_imputer.fit(log_transformed_data)
    imputed_data = tsvd_imputer.inverse_transform(tsvd_imputer.transform(log_transformed_data))

# 步骤 4: 按行进行标准化
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(imputed_data)

# 步骤 5: 每列数据减去对应的中位数
    column_medians = np.median(normalized_data, axis=0)
    median_subtracted_data = normalized_data - column_medians

# 步骤 6: 使用tSVD将数据降至128维
    tsvd = TruncatedSVD(n_components=128)
    reduced_data = tsvd.fit_transform(median_subtracted_data)

# 生成列名
    column_names = [f'component_{i}' for i in range(128)]
    reduced_data_df = pd.DataFrame(reduced_data, index=df_multi_target_data.index, columns=column_names)

# 打印降维后的数据
    

# 保存降维后的数据和相关信息
    with tables.File(final_multi_target_hdf5_path, mode='w') as hdf_file:
    # 保存降维数据
        hdf_file.create_carray(
            where='/',
            name='reduced_data',
            obj=reduced_data_df.to_numpy(),
            chunkshape=(100, reduced_data_df.shape[1]),
            filters=tables.Filters(complevel=9)
        )
    
    # 保存列名
        hdf_file.create_array(
            where='/',
            name='original_columns',
            obj=np.array(column_names, dtype='S')
        )
    
    # 保存行标签
        row_labels = reduced_data_df.index.tolist()
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

    # 保存tSVD插补所需的参数
        hdf_file.create_array(
            where='/',
            name='tsvd_imputer_components',
            obj=tsvd_imputer.components_
        )

    print()
    print("reduced_targe_data:")
    print(reduced_data_df.shape)
    print(reduced_data_df)






input_preprocessing()
target_preprocessing()