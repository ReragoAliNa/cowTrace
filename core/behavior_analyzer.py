import numpy as np
import cv2
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class SingleCattleData:
    """定义单只牛的多任务预测数据结构"""
    cattle_id: int                    # 跟踪ID
    bbox: Tuple[int, int, int, int]    # 目标检测框 (x_min, y_min, width, height)
    mask: np.ndarray                  # 实例分割掩膜 (H, W 的二值矩阵)
    keypoints: np.ndarray             # 姿态估计关键点，形状为 (num_kpts, 3) -> [x, y, confidence]

class CattleBehaviorAnalyzer:
    def __init__(self, fps: int = 25):
        self.fps = fps
        # 维护牛只的历史轨迹数据，Key为 cattle_id, Value为历史质心坐标的 List
        self.trajectory_history: Dict[int, List[Tuple[int, int]]] = {}
        # 维护牛只的躺卧累计时间（帧数）
        self.lying_frames: Dict[int, int] = {}
        
        # 业务逻辑相关阈值（可写在 config.py 中）
        self.LYING_RATIO_THRESH = 1.5    # 躺卧时外接矩形宽高比阈值（宽/高）
        self.LYING_HEIGHT_PCT = 0.4      # 躺卧时高度占其站立平均高度的比例
        self.BACK_ARCH_THRESH = 15.0     # 判定跛行的背部弯曲度阈值（角度）

    def update(self, frame_data: List[SingleCattleData]) -> Dict[int, Dict]:
        """
        核心更新接口：每一帧调用一次，输入当前帧所有牛的数据，输出异常诊断结果
        """
        results = {}
        active_ids = []

        for cattle in frame_data:
            c_id = cattle.cattle_id
            active_ids.append(c_id)
            
            # 1. 运动轨迹分析
            centroid = self._analyze_trajectory(c_id, cattle.bbox)
            
            # 2. 躺卧行为判定
            is_lying, lying_duration = self._analyze_lying(c_id, cattle.bbox, cattle.mask)
            
            # 3. 跛行行为诊断（基于姿态关键点）
            is_lame, back_angle = self._analyze_lameness(cattle.keypoints)
            
            # 汇总当前牛只的状态
            results[c_id] = {
                "centroid": centroid,
                "is_lying": is_lying,
                "lying_duration_sec": lying_duration,
                "is_lame": is_lame,
                "back_angle": back_angle
            }
            
        # 清理已经离开画面的牛只历史数据，防止内存泄露
        self._clear_dead_tracks(active_ids)
        return results

    def _analyze_trajectory(self, cattle_id: int, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        【图6 - 运动轨迹 映射 目标检测】
        通过目标检测框计算质心，并在时间序列上累加形成轨迹。
        """
        x, y, w, h = bbox
        centroid = (int(x + w / 2), int(y + h / 2))
        
        if cattle_id not in self.trajectory_history:
            self.trajectory_history[cattle_id] = []
        
        self.trajectory_history[cattle_id].append(centroid)
        
        # 仅保留最近 10 秒（250帧）的轨迹用于计算速度或绘制轨迹线
        if len(self.trajectory_history[cattle_id]) > self.fps * 10:
            self.trajectory_history[cattle_id].pop(0)
            
        return centroid

    def _analyze_lying(self, cattle_id: int, bbox: Tuple[int, int, int, int], mask: np.ndarray) -> Tuple[bool, float]:
        """
        【图6 - 躺卧行为 映射 实例分割】
        结合实例分割掩膜和边界框比例，判定牛只是否处于躺卧状态。
        """
        _, _, w, h = bbox
        if h == 0:
            return False, 0.0
            
        aspect_ratio = w / h
        
        # 躺卧判定逻辑：宽高比大（身体横向拉长）且高度较矮
        # 在实际工程中，还可以利用 mask 的面积大小和 y 轴重心来辅助判定
        is_lying_state = False
        if aspect_ratio > self.LYING_RATIO_THRESH:
            # 进一步通过掩膜（Mask）验证其几何紧凑度
            mask_area = np.sum(mask > 0)
            box_area = w * h
            solidity = mask_area / box_area if box_area > 0 else 0
            # 躺卧时牛只身体通常蜷缩，solidity (实心度) 会偏高
            if solidity > 0.6: 
                is_lying_state = True
        
        # 统计持续时间
        if cattle_id not in self.lying_frames:
            self.lying_frames[cattle_id] = 0
            
        if is_lying_state:
            self.lying_frames[cattle_id] += 1
        else:
            self.lying_frames[cattle_id] = 0 # 若起立则重置
            
        lying_seconds = self.lying_frames[cattle_id] / self.fps
        return is_lying_state, lying_seconds

    def _analyze_lameness(self, keypoints: np.ndarray) -> Tuple[bool, float]:
        """
        【图6 - 跛行行为 映射 姿态估计】
        利用姿态估计输出的脊椎、肩部、臀部关键点，计算背部弯曲度（Arch Angle）。
        
        假设关键点 index 定义：
        0: 头部, 1: 脖子, 2: 肩部(Shoulder), 3: 脊椎中点(Spine_mid), 4: 臀部(Hip)
        """
        # 如果关键点置信度太低，不进行诊断（这里假设第3列为置信度）
        required_indices = [2, 3, 4]
        for idx in required_indices:
            if idx >= len(keypoints) or keypoints[idx][2] < 0.5:
                return False, 0.0  # 置信度不足，跳过
                
        # 提取肩、脊椎中点、臀的三维/二维坐标 (x, y)
        p_shoulder = keypoints[2][:2]
        p_spine = keypoints[3][:2]
        p_hip = keypoints[4][:2]
        
        # 计算肩到臀的向量，以及脊椎中点偏离该直线的垂直距离
        # 跛行牛行走时由于疼痛，会弓背（Back Arching），导致脊椎中点 y 坐标向上偏移
        # 计算向量夹角：肩-脊椎 向量 与 脊椎-臀 向量 之间的夹角
        v1 = p_shoulder - p_spine
        v2 = p_hip - p_spine
        
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_degree = np.degrees(angle)
        
        # angle_degree 越接近 180 度代表背部越直，越小代表背部拱起越严重
        # 实际跛行指标：计算拱起角度
        arch_angle = 180.0 - angle_degree
        is_lame = arch_angle > self.BACK_ARCH_THRESH
        
        return is_lame, arch_angle

    def _clear_dead_tracks(self, active_ids: List[int]):
        """清理已消失牛只的缓存，释放内存"""
        all_tracked_ids = list(self.trajectory_history.keys())
        for t_id in all_tracked_ids:
            if t_id not in active_ids:
                del self.trajectory_history[t_id]
                if t_id in self.lying_frames:
                    del self.lying_frames[t_id]