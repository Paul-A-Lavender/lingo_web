###
#  用来放一些非展示界面的代码
###

from typing import List
import numpy as np
from scipy.ndimage import binary_fill_holes
import trimesh
import numpy as np
from  trimesh.voxel.creation import voxelize

def voxelize(path:str,output:str="./cache/default.npy")->np.ndarray:
    '''
    “体素化”，类似于把矢量图“像素化”，是把一个三维模型采样为一个三维数组，每个元素描述对应位置的“体素”是否与模型重合
    从指定路径下读取场景文件，并将其采样，转化为高维数组，存储至输入数据缓存区（./cache）
    @param path:str 场景文件的路径
    @param output:str="./cache/default.npy" 存储.npy文件的路径和文件名
    @return:np.ndarray 高维np数组的转化结果
    '''

    # 读取 .obj 文件
    mesh = trimesh.load_mesh(path).to_mesh()

    target_shape = (400, 100, 600)

    # 计算模型的长宽高
    bounds_min, bounds_max = mesh.bounds
    model_size = bounds_max - bounds_min  # 长宽高

    # 计算每个维度的缩放因子，保持长宽高比
    scaling_factors = np.array(target_shape) / model_size

    # 取最小的缩放因子，以保证场景尺寸不超过目标尺寸
    scaling_factor = min(scaling_factors)
    # 目标体素尺寸

    # 计算模型边界范围
    bounds_min, bounds_max = mesh.bounds
    model_size = bounds_max - bounds_min

    # 计算缩放比例，使模型正好适配目标体素网格
    scaling_factors = np.array(target_shape) / model_size
    scaling_factor = min(scaling_factors)*0.995  # 保证模型按最小比例等比缩放

    # 缩放模型
    mesh.apply_scale(scaling_factor)
    # 平移模型到体素网格中心
    mesh.apply_translation(-mesh.bounds[0])  # 移动到原点
    mesh.apply_translation([0,0.2,0])  # 移动到原点
    grid_size = np.array(target_shape)
    mesh.apply_translation(grid_size / 2 - model_size * scaling_factor / 2)

    # # 体素化模型
    # 将网格转换为体素网格
    voxel_grid = voxelize(mesh,pitch=1) # pitch是体素的大小

    # 获取体素网格的布尔值表示
    voxel_grid = voxel_grid.matrix
    np.save(output, voxel_grid)
    return voxel_grid
        
    
def prep_lingo_job(start_locations:List,end_locations:List,actions:List,output:str="./queue")->bool:
    '''
    接收模型的起始/终止位置以及行动提示词，并整理模型输入至“./queue”，等待作业队列系统处理。
    @param start_locations:List 起始坐标列表
    @param start_locations:List 终止坐标列表
    @param start_locations:List 行动提示词列表
    @return:bool 操作是否成功（相关输入是否齐全）
    '''
    assert len(start_locations)==len(end_locations)==len(actions) # 三项输入应当等长


###############################################################
#               以下为工具函数（并非接口）                      #
###############################################################

def fill_voxel_matrix(original_matrix):
    """
    将体素矩阵的内部区域填充为 True。
    假设 original_matrix 的 True 表示外壳。
    """
    # 使用 binary_fill_holes 对三维矩阵进行内部填充
    filled_matrix = binary_fill_holes(original_matrix)
    
    return filled_matrix


def pad_voxel_matrix_with_y_padding(original_matrix, target_shape):
    """
    将体素矩阵填充到目标尺寸，y轴的padding拼接在原矩阵的末尾，其余维度居中填充。
    """
    # 获取原始矩阵的形状
    original_shape = np.array(original_matrix.shape)
    target_shape = np.array(target_shape)

    # 计算需要填充的数量
    padding = target_shape - original_shape

    # 分别计算 x 和 z 轴（对称填充）和 y 轴（只在末尾填充）的填充量
    x_padding = padding[0] // 2
    z_padding = padding[2] // 2
    y_padding_start = 0
    y_padding_end = padding[1]

    # 构造全False的目标矩阵
    padded_matrix = np.zeros(target_shape, dtype=bool)

    # 构造切片：x 和 z 居中填充，y 在末尾拼接
    slices = (
        slice(x_padding, x_padding + original_shape[0]),  # x 轴填充
        slice(0, original_shape[1]),                      # y 轴不填充开始部分
        slice(z_padding, z_padding + original_shape[2])   # z 轴填充
    )

    # 将原始矩阵放置到目标矩阵中
    padded_matrix[slices] = original_matrix

    return padded_matrix