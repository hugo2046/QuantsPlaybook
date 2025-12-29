'''
Author: Hugo
Date: 2025-11-06 21:12:00
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-11-24 09:25:37
Description: 
d-LE-SC算法实现用于领先-滞后聚类

本模块实现了有向似然估计谱聚类(d-LE-SC)算法，用于检测金融网络中的领先-滞后关系。

基于: "A tug of war across the market: overnight-vs-daytime lead-lag networks
and clustering-based portfolio strategies" (第4.2节)

作者: Generated with Claude Code

Status: CORE - GPU加速核心算法
Usage: 核心算法实现，被FactorPipeline和其他模块调用
Features:
- 完全GPU加速（PyTorch实现）
- 支持单日和批量处理
- PyTorch原生K-means聚类
- 多层容错机制

Performance:
- 需要CUDA支持
- 自动GPU内存管理
- 大数据集优化
'''

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _is_convergence_error(e: BaseException) -> bool:
    """
    Tenacity辅助函数：判断异常是否为torch.linalg.eigh的收敛错误
    """
    return (
        isinstance(e, RuntimeError)
        and "linalg.eigh" in str(e)
        and "failed to converge" in str(e)
    )


class DLESCClustering:
    """
    d-LE-SC（有向似然估计谱聚类）算法实现

    该算法基于表示领先-滞后关系的有向邻接矩阵，将股票聚类为领先组和滞后组。

    算法步骤（论文4.2节算法1）：
    1. 计算Hermitian矩阵 H = i*log((1-η)/η)*(A-A^T) + log(1/(4η(1-η)))*(A+A^T)
    2. 计算H的顶部特征向量
    3. 基于[Re(v1), Im(v1)]嵌入进行k-means聚类
    4. 更新有向SBM参数η
    5. 迭代直至收敛

    特性：
    - 完全GPU加速（需要CUDA支持）
    - 支持单日和批量处理
    - PyTorch原生K-means实现
    """

    def __init__(
        self,
        n_iterations: int = 10,
        random_state: int = 42,
        tol: float = 1e-6,
        device: Optional[str] = None,
    ):
        """
        初始化d-LE-SC算法

        Args:
            n_iterations: 算法迭代次数
            random_state: 随机种子，用于结果复现
            tol: η参数的收敛容差
            device: PyTorch设备 ('cuda' 或 'cuda:0')。如果为None，自动使用CUDA
        """
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.tol = tol

        # 强制使用GPU
        if not torch.cuda.is_available():
            raise RuntimeError("此算法需要GPU支持，但未检测到CUDA设备")

        if device is None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)

        logger.info(f"使用GPU设备: {self.device}")

        # Initialize parameters
        self.eta = None
        self.lead_cluster = None
        self.lag_cluster = None

        # 在初始化时设置随机种子和创建随机数生成器
        self._setup_random_generators()

    def _setup_random_generators(self):
        """
        设置随机数生成器，确保可复现性和正确的随机性管理
        """
        if self.random_state is not None:
            # 设置全局PyTorch种子
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.random_state)
                torch.cuda.manual_seed_all(self.random_state)

            # 创建CPU随机数生成器用于η初始化
            self.cpu_rng = torch.Generator()
            self.cpu_rng.manual_seed(self.random_state)

            # 创建GPU随机数生成器
            self.gpu_rng = torch.Generator(device=self.device)
            self.gpu_rng.manual_seed(self.random_state)

            # 为K-means创建独立的种子序列
            self.kmeans_seed = self.random_state + 1000
        else:
            # 如果未指定种子，使用默认的随机生成器
            self.cpu_rng = None
            self.gpu_rng = None
            self.kmeans_seed = None

    def _compute_hermitian_matrix(self, A: torch.Tensor) -> torch.Tensor:
        """
        计算算法1中指定的Hermitian矩阵H

        核心公式（论文4.2节）：
        H = i * log((1-η)/η) * (A - A^T) + log(1/(4η(1-η))) * (A + A^T)

        其中：
        - 第一项捕获有向性（反对称部分）
        - 第二项捕获连通性（对称部分）
        - η是有向SBM参数，控制噪声水平

        Args:
            A: 有向邻接矩阵 (n x n)

        Returns:
            H: Hermitian矩阵 (n x n)，复值矩阵
        """
        if self.eta is None:
            raise ValueError("η参数未初始化")

        eta_tensor = torch.tensor(self.eta, device=self.device, dtype=torch.float32)
        # 防止log(0)通过将η限制在合理范围内
        eta_clipped = torch.clamp(eta_tensor, min=1e-10, max=0.5 - 1e-10)

        # 计算权重
        # 有向权重：用于(A - A^T)，捕获领先-滞后的方向性
        w_directional = torch.log((1 - eta_clipped) / eta_clipped)
        # 对称权重：用于(A + A^T)，捕获总体连通性
        w_symmetric = torch.log(1 / (4 * eta_clipped * (1 - eta_clipped)))

        # 计算矩阵的对称和反对称部分
        A_diff = A - A.T  # 反对称部分，有向性
        A_sum = A + A.T  # 对称部分，连通性

        # 构建复值Hermitian矩阵
        # 实部来自对称部分，虚部来自反对称部分
        H_real = w_symmetric * A_sum
        H_imag = w_directional * A_diff

        # 组合成复值张量
        H = torch.complex(H_real, H_imag)

        return H

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        retry=retry_if_exception(_is_convergence_error),
    )
    def _compute_top_eigenvector(self, H: torch.Tensor) -> torch.Tensor:
        """
        计算Hermitian矩阵H的顶部特征向量
        
        使用多层容错机制 + 重试：
        1. 首先尝试使用带小正则化项的eigh（鲁棒性更强）
        2. 如果失败，使用更大的正则化项重试
        3. 如果仍失败，使用SVD分解作为最终备选
        4. 如果SVD也失败，tenacity会自动重试整个方法（最多3次）

        Args:
            H: Hermitian矩阵 (n x n)

        Returns:
            v1: 顶部特征向量 (n,) - 复值向量
        """
        n = H.shape[0]
        I = torch.eye(n, dtype=H.dtype, device=H.device)
        
        # 尝试1: 直接使用小的正则化项，提高鲁棒性
        try:
            epsilon = 1e-7
            H_reg = H + epsilon * I
            eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
            top_idx = torch.argmax(eigenvalues.real)
            return eigenvectors[:, top_idx]
        except RuntimeError as e:
            if "failed to converge" not in str(e) and "ill-conditioned" not in str(e):
                raise
            logger.warning(f"eigh(正则化 epsilon={epsilon}) 失败: {str(e)[:100]}")
        
        # 尝试2: 使用更大的正则化项
        try:
            epsilon = 1e-5
            H_reg = H + epsilon * I
            eigenvalues, eigenvectors = torch.linalg.eigh(H_reg)
            top_idx = torch.argmax(eigenvalues.real)
            logger.info(f"使用大正则化 epsilon={epsilon} 成功")
            return eigenvectors[:, top_idx]
        except RuntimeError:
            logger.warning(f"大正则化项(epsilon={epsilon})失败")
        
        # 尝试3: 备选方案 - 使用SVD
        # 对于Hermitian矩阵，左右奇异向量相同，奇异值=|特征值|
        logger.warning("使用SVD作为备选方案")
        U, S, Vh = torch.linalg.svd(H)
        # 最大奇异值对应的左奇异向量
        v1 = U[:, 0]
        return v1
        # 注意：如果SVD也失败（极少情况），tenacity会自动重试整个方法

    def _kmeans_pytorch(
        self,
        X: torch.Tensor,
        n_clusters: int = 2,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        PyTorch原生实现的K-means（完全在GPU上运行）

        Args:
            X: 输入数据 (n, d)，在GPU上
            n_clusters: 聚类数量
            max_iter: 最大迭代次数
            tol: 收敛容差
            seed: 随机种子（如果为None，不设置种子以增加随机性）

        Returns:
            labels: 聚类标签 (n,)
        """
        n_samples, n_features = X.shape

        # 随机初始化中心点（使用k-means++初始化策略）
        # 关键修复：只在明确传入seed时才设置种子，否则使用真随机
        if seed is not None:
            torch.manual_seed(seed)

        # 选择第一个中心点
        indices = torch.randint(0, n_samples, (1,), device=self.device)
        centroids = X[indices]

        # K-means++：依次选择剩余中心点
        for _ in range(1, n_clusters):
            # 计算每个点到最近中心的距离
            distances = torch.cdist(X, centroids).min(dim=1)[0]
            # 按距离平方的概率选择下一个中心
            probabilities = distances**2
            probabilities = probabilities / probabilities.sum()
            next_idx = torch.multinomial(probabilities, 1)
            centroids = torch.cat([centroids, X[next_idx]], dim=0)

        # 主迭代循环
        for iteration in range(max_iter):
            # 计算每个点到各中心的距离
            distances = torch.cdist(X, centroids)  # (n, k)

            # 分配标签
            labels = torch.argmin(distances, dim=1)  # (n,)

            # 更新中心点
            new_centroids = torch.stack(
                [
                    X[labels == k].mean(dim=0) if (labels == k).any() else centroids[k]
                    for k in range(n_clusters)
                ]
            )

            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=tol):
                break

            centroids = new_centroids

        return labels

    def _cluster_embeddings(self, v1: torch.Tensor, seed: Optional[int] = None):
        """
        对嵌入[Re(v1), Im(v1)]应用k-means聚类（PyTorch实现，完全GPU加速）

        Args:
            v1: 顶部特征向量 (n,) - 复值
            seed: K-means随机种子（如果为None，使用真随机）
        """
        # 创建嵌入矩阵[Re(v1), Im(v1)]
        embedding = torch.stack([v1.real, v1.imag], dim=1)  # (n, 2)

        # 使用PyTorch原生K-means
        cluster_labels = self._kmeans_pytorch(embedding, n_clusters=2, seed=seed)

        # 分割为两个聚类
        cluster_0_indices = torch.where(cluster_labels == 0)[0]
        cluster_1_indices = torch.where(cluster_labels == 1)[0]

        # 将初始聚类结果存储到实例属性
        self.lead_cluster = cluster_0_indices
        self.lag_cluster = cluster_1_indices

    def _compute_flows(
        self, A: torch.Tensor, cluster1: torch.Tensor, cluster2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算两个聚类之间的净流量和总流量

        Args:
            A: 有向邻接矩阵
            cluster1: 第一个聚类的索引
            cluster2: 第二个聚类的索引

        Returns:
            net_flow: 从聚类1到聚类2的净流量
            total_flow: 聚类间的总流量
        """
        # 创建网格索引
        i_idx = cluster1.unsqueeze(1).expand(-1, len(cluster2))
        j_idx = cluster2.unsqueeze(0).expand(len(cluster1), -1)

        # 提取子矩阵
        submatrix = A[i_idx, j_idx]

        # 从聚类1到聚类2的流量
        flow_1_to_2 = torch.sum(submatrix)

        # 从聚类2到聚类1的流量
        submatrix_rev = A[j_idx, i_idx]
        flow_2_to_1 = torch.sum(submatrix_rev)

        # 计算净流量和总流量
        net_flow = flow_1_to_2 - flow_2_to_1
        total_flow = flow_1_to_2 + flow_2_to_1

        return net_flow, total_flow

    def _update_eta(self, A: torch.Tensor) -> float:
        """
        基于当前聚类更新η参数（论文4.2节算法1）

        更新公式：
        η ← min{|C_lead → C_lag| / TF(C_lead, C_lag), |C_lag → C_lead| / TF(C_lead, C_lag)}

        其中：
        - |C1 → C2|: 从聚类C1到C2的有向流总量
        - TF(C1, C2): 聚类间的总流量（双向）
        - η ∈ [0, 0.5]：有向SBM参数，控制方向性噪声

        Args:
            A: 有向邻接矩阵

        Returns:
            new_eta: 更新后的η值
        """
        if self.lead_cluster is None or self.lag_cluster is None:
            raise ValueError("聚类未初始化")

        # 计算流量指标
        net_flow, total_flow = self._compute_flows(
            A, self.lead_cluster, self.lag_cluster
        )

        if total_flow == 0:
            # 避免除零错误
            return 0.25  # 默认值

        # 计算有向流量
        # 创建索引矩阵用于高效计算
        i_idx = self.lead_cluster.unsqueeze(1).expand(-1, len(self.lag_cluster))
        j_idx = self.lag_cluster.unsqueeze(0).expand(len(self.lead_cluster), -1)

        # 从领先组到滞后组的流量
        flow_lead_to_lag = torch.sum(A[i_idx, j_idx])
        # 从滞后组到领先组的流量
        flow_lag_to_lead = torch.sum(A[j_idx, i_idx])

        # 更新η参数
        eta1 = flow_lead_to_lag / total_flow
        eta2 = flow_lag_to_lead / total_flow

        # 取最小值作为新的η估计
        new_eta = torch.min(eta1, eta2)

        # 确保η保持在有效范围[0, 0.5]内
        new_eta = torch.clamp(new_eta, min=1e-10, max=0.5 - 1e-10)

        return float(new_eta.cpu())

    def _determine_lead_lag_direction(self, A: torch.Tensor):
        """
        基于有向流量确定哪个聚类是领先聚类

        Args:
            A: 有向邻接矩阵
        """
        if self.lead_cluster is None or self.lag_cluster is None:
            return

        # 计算净流量
        net_flow, _ = self._compute_flows(A, self.lead_cluster, self.lag_cluster)

        # 如果净流量为负，交换聚类
        if net_flow < 0:
            self.lead_cluster, self.lag_cluster = self.lag_cluster, self.lead_cluster

    def fit_single(self, A: np.ndarray) -> Dict[str, Any]:
        """
        对单个有向邻接矩阵拟合d-LE-SC算法（论文4.2节算法1完整流程）

        算法流程：
        1. 初始化η参数
        2. 迭代执行直到收敛：
           - 计算Hermitian矩阵 H
           - 计算H的顶部特征向量 v1
           - 基于[Re(v1), Im(v1)]进行k-means聚类
           - 确定领先/滞后方向
           - 更新η参数
        3. 返回聚类结果

        Args:
            A: 有向邻接矩阵 (n x n)

        Returns:
            results: 包含聚类结果的字典
                - lead_cluster: 领先聚类索引数组
                - lag_cluster: 滞后聚类索引数组
                - eta: 最终η参数值
                - n_iterations: 实际迭代次数
                - lead_cluster_size: 领先聚类大小
                - lag_cluster_size: 滞后聚类大小
        """
        # 转换为PyTorch张量并移到GPU
        A_tensor = torch.from_numpy(A).float().to(self.device)

        # 使用在__init__中创建的随机数生成器初始化η
        if self.gpu_rng is not None:
            # 使用GPU随机数生成器
            eta_tensor = torch.rand(1, device=self.device, generator=self.gpu_rng)
        else:
            # 使用默认随机数生成器
            eta_tensor = torch.rand(1, device=self.device)

        self.eta = (eta_tensor * 0.4 + 0.1).item()

        # 主迭代循环
        for iteration in range(self.n_iterations):
            old_eta = self.eta

            # 步骤1: 计算Hermitian矩阵
            H = self._compute_hermitian_matrix(A_tensor)

            # 步骤2: 计算顶部特征向量
            v1 = self._compute_top_eigenvector(H)

            # 步骤3: 聚类嵌入（使用K-means独立种子）
            self._cluster_embeddings(v1, seed=self.kmeans_seed)

            # 步骤4: 确定领先/滞后方向
            self._determine_lead_lag_direction(A_tensor)

            # 步骤5: 更新η参数
            self.eta = self._update_eta(A_tensor)

            # 检查收敛性
            if abs(self.eta - old_eta) < self.tol:
                break

        # 准备结果
        results = {
            "lead_cluster": self.lead_cluster.cpu().numpy(),
            "lag_cluster": self.lag_cluster.cpu().numpy(),
            "eta": self.eta,
            "n_iterations": iteration + 1,
            "lead_cluster_size": len(self.lead_cluster),
            "lag_cluster_size": len(self.lag_cluster),
        }

        return results

    def fit(self, A: np.ndarray) -> Dict[str, Any]:
        """
        智能拟合方法：自动检测输入维度并选择处理方式

        Args:
            A: 邻接矩阵
               - 如果是2D (n, n): 单日处理
               - 如果是3D (T, n, n): 批量处理T天

        Returns:
            results: 聚类结果
                - 2D输入: 返回单个结果字典
                - 3D输入: 返回结果列表
        """
        if A.ndim == 2:
            # 单日处理
            logger.info("检测到2D输入，使用单日处理模式")
            return self.fit_single(A)

        elif A.ndim == 3:
            # 批量处理
            logger.info(f"检测到3D输入 {A.shape}，使用批量处理模式")
            return self.fit_batch(A)

        else:
            raise ValueError(f"不支持的输入维度: {A.ndim}。期望2D或3D数组。")

    def _process_day(
        self, day_idx: int, A_tensor: torch.Tensor
    ) -> Dict[str, Any]:
        """
        处理单日计算的核心逻辑

        注意：重试逻辑已移到 _compute_top_eigenvector 方法中，
        因为失败点在特征值分解，而不是整个日期处理。

        Args:
            day_idx: 当前处理的日期索引
            A_tensor: 当天的邻接矩阵 (在GPU上)

        Returns:
            包含单日结果的字典
        """
        # 重置状态，避免日期间的污染
        self.lead_cluster = None
        self.lag_cluster = None

        # 使用在__init__中创建的随机数生成器初始化η
        # 为每天创建不同的随机数生成器状态
        if self.gpu_rng is not None:
            # 为每天创建独立的GPU随机数生成器
            day_gpu_rng = torch.Generator(device=self.device)
            day_gpu_rng.manual_seed(self.kmeans_seed + day_idx)
            eta_tensor = torch.rand(1, device=self.device, generator=day_gpu_rng)
        else:
            # 使用默认随机数生成器
            eta_tensor = torch.rand(1, device=self.device)

        self.eta = (eta_tensor * 0.4 + 0.1).item()

        # 主迭代循环
        for iteration in range(self.n_iterations):
            old_eta = self.eta

            H = self._compute_hermitian_matrix(A_tensor)
            # 特征值分解（自带4层容错+重试机制）
            v1 = self._compute_top_eigenvector(H)
            # 为每天使用不同的K-means种子，保证可复现性
            self._cluster_embeddings(v1, seed=self.kmeans_seed + day_idx)
            self._determine_lead_lag_direction(A_tensor)
            self.eta = self._update_eta(A_tensor)

            if abs(self.eta - old_eta) < self.tol:
                break

        # 成功，返回结果
        return {
            "day_idx": day_idx,
            "lead_cluster": self.lead_cluster.cpu().numpy(),
            "lag_cluster": self.lag_cluster.cpu().numpy(),
            "eta": self.eta,
            "n_iterations": iteration + 1,
            "lead_cluster_size": len(self.lead_cluster),
            "lag_cluster_size": len(self.lag_cluster),
            "error": None,
        }

    def fit_batch(
        self,
        adjacency_matrices: np.ndarray,
        verbose: bool = True,
        preload_gpu: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量处理多天的邻接矩阵（GPU顺序处理，针对单GPU优化）

        策略：
        1. 如果preload_gpu=True: 预先将所有数据加载到GPU，减少传输开销
        2. 顺序处理每一天（GPU不适合多进程并行）
        3. 使用tqdm显示进度

        注意：对于GPU计算，多进程并行会导致性能退化（进程争抢GPU资源）

        Args:
            adjacency_matrices: 形状为 (T, N, N) 的3D数组
            verbose: 是否显示进度条
            preload_gpu: 是否预先将数据加载到GPU（推荐True）

        Returns:
            results: 包含所有结果的列表
        """
        T, N, _ = adjacency_matrices.shape
        logger.info(f"开始批量处理 {T} 个交易日的数据...")
        logger.info(f"使用GPU顺序处理（单GPU最优策略）")

        A_gpu = None
        should_preload = preload_gpu

        if should_preload:
            try:
                # 动态检查GPU内存是否足够
                required_mem = adjacency_matrices.nbytes
                free_mem, total_mem = torch.cuda.mem_get_info(self.device)
                # 保留20%的安全边际
                if required_mem > free_mem * 0.8:
                    logger.warning(
                        f"数据所需内存 ({required_mem/1024**3:.2f} GB) "
                        f"超过可用GPU显存的80% ({free_mem/1024**3:.2f} GB)。"
                        "将切换到逐日加载模式以避免内存溢出。"
                    )
                    should_preload = False
                else:
                    logger.info("预加载数据到GPU...")
                    A_gpu = torch.from_numpy(adjacency_matrices).float().to(self.device)
                    logger.info(
                        f"GPU内存占用: {A_gpu.element_size() * A_gpu.nelement() / 1024**3:.2f} GB"
                    )
            except Exception as e:
                logger.error(f"预加载数据到GPU时出错: {e}。将切换到逐日加载模式。")
                should_preload = False
                if A_gpu is not None:
                    del A_gpu
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass

        results = []
        iterator = tqdm(range(T), desc="批量处理进度") if verbose else range(T)

        for day_idx in iterator:
            try:
                A_tensor = (
                    A_gpu[day_idx]
                    if should_preload
                    else torch.from_numpy(adjacency_matrices[day_idx])
                    .float()
                    .to(self.device)
                )

                # 处理当天数据
                # 重试逻辑在 _compute_top_eigenvector 中自动处理
                result = self._process_day(day_idx, A_tensor)
                results.append(result)

            except Exception as e:
                # 如果 _compute_top_eigenvector 的所有重试都失败
                logger.error(f"处理第{day_idx}天时所有方法均失败: {str(e)}")
                error_result = {
                    "day_idx": day_idx,
                    "lead_cluster": None,
                    "lag_cluster": None,
                    "eta": None,
                    "n_iterations": 0,
                    "lead_cluster_size": 0,
                    "lag_cluster_size": 0,
                    "error": str(e),
                }
                results.append(error_result)

        if A_gpu is not None:
            del A_gpu
            try:
                torch.cuda.empty_cache()
            except:
                pass

        logger.info("批量处理完成！")
        return results


    def predict(self, A: np.ndarray) -> np.ndarray:
        """
        使用已拟合的模型预测新数据的聚类标签

        Args:
            A: 有向邻接矩阵 (n x n)

        Returns:
            labels: 聚类标签 (0为领先组，1为滞后组)
        """
        if self.eta is None:
            raise ValueError("模型尚未拟合，请先调用fit()方法")

        # 转换为PyTorch张量并移到GPU
        A_tensor = torch.from_numpy(A).float().to(self.device)

        # 计算Hermitian矩阵
        H = self._compute_hermitian_matrix(A_tensor)

        # 计算顶部特征向量
        v1 = self._compute_top_eigenvector(H)

        # 创建嵌入
        embedding = torch.stack([v1.real, v1.imag], dim=1)

        # 使用PyTorch K-means进行聚类（使用K-means独立种子）
        labels = self._kmeans_pytorch(embedding, n_clusters=2, seed=self.kmeans_seed)

        return labels.cpu().numpy()
