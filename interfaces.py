
import numpy as np
import trimesh
import datetime
from utils import *


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
        
    
def prep_lingo_job(inference_info:inference_info,output:str="./queue",overwrite:bool=True,submitted_time:datetime.datetime=None)->str:
    '''
    接收模型的起始/终止位置以及行动提示词，并整理模型输入至“./queue”，等待作业队列系统处理。
    假设场景文件为livingroom.obj，上传时间为2025/01/22 14：24，则/queue下目录结构为：
        /livingroom-2025-01-22-14-24
            job_description.json:{
                start_locations:[...],
                end_locations:[...],
                actions:[...],
                submitted_time:2025-01-22 14:41:48.679491,
                scene_name:"livingroom"
            }
            livingroom.obj
            livingroom_voxelized.npy
    
    @param start_locations:List 起始坐标列表
    @param end_locations:List 终止坐标列表
    @param actions:List 行动提示词列表
    @param scene_path:str 场景文件路径
    @param overwrite:bool=True 若对应目录下已存在同名文件，是否覆写(如否则会在发现重名时退出)
    @param submitted_time=None 提交时间,若之前已提交则可使用之前的提交时间，否则按当前时间新建
    @return:bool 操作是否成功（相关输入是否齐全）
    '''
    assert inference_info.is_valid() # 验证数据有效性
    current_time = datetime.datetime.now()