# -- coding: utf-8 --
from config import *


class FeedForwardNet_BN_LeakyDropout(nn.Module):
    """
    前馈神经网络（带 Batch Normalization、LeakyReLU 和 Dropout）

    该网络主要用于回归任务，在隐藏层中先使用全连接层，然后依次应用：
      - Batch Normalization：归一化每一层输出，缓解内部协变量偏移。
      - LeakyReLU 激活：允许负值有很小的梯度，避免 ReLU 的“神经元死亡”问题。
      - Dropout：随机丢弃一部分神经元输出，减少过拟合。

    输出层为线性层（无激活），适用于回归任务。
    """

    def __init__(self, input_dim, layers, dropout_rate=0):
        """
        :param input_dim: 输入特征数
        :param layers: 每层神经元数的列表，其中最后一项为输出层神经元数
        :param dropout_rate: Dropout 的丢弃率，默认为 0.2
        """
        super().__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim

        # 构建隐藏层（最后一层为输出层）
        for i, num_units in enumerate(layers[:-1]):
            # 全连接层：将输入维度映射到当前层神经元个数
            self.layers.append(nn.Linear(prev_dim, num_units))
            
            # 批标准化：使得每个批次中该层输出均值为 0、方差为 1
            # 多次仿真证实，BN对浅层网络并无好处，一般其在训练深层网络有效果
            # self.layers.append(nn.BatchNorm1d(num_units))
            
            # LeakyReLU 激活：负半轴以 0.2 的斜率输出
            self.layers.append(nn.LeakyReLU())
            
            # Dropout 正则化：随机丢弃部分输出，防止过拟合
            self.layers.append(nn.Dropout(dropout_rate))
            prev_dim = num_units

        # 输出层：线性映射到输出维度（回归任务通常不使用激活函数）
        self.layers.append(nn.Linear(prev_dim, layers[-1]))

    def forward(self, x):
        """
        前向传播：依次将输入 x 通过所有层，最后输出预测结果
        """
        # 逐层计算隐藏层输出
        for layer in self.layers[:-1]:
            x = layer(x)
        # 输出层直接计算，不加激活函数
        return self.layers[-1](x)



class FeedForwardNet_ReLU(nn.Module):
    """
    前馈神经网络（标准全连接网络），隐藏层使用 ReLU 激活

    该网络用于回归或分类任务，每个隐藏层使用 ReLU 激活函数，
    输出层直接输出预测结果（回归任务通常不使用激活函数）。
    """

    def __init__(self, input_dim, layers):
        """
        :param input_dim: 输入特征数
        :param layers: 每层神经元数的列表，其中最后一项为输出层神经元数
        """
        super(FeedForwardNet_ReLU, self).__init__()
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        # 添加每一层的全连接层
        for num_units in layers:
            self.layers.append(nn.Linear(prev_dim, num_units))
            prev_dim = num_units

    def forward(self, x):
        """
        前向传播：隐藏层采用 ReLU 激活，输出层直接输出预测结果
        """
        # 对除输出层之外的每一层应用 ReLU 激活
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        # 输出层直接输出（回归任务无激活）
        return self.layers[-1](x)





class CombinedRegClsModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=16, bn_dim_reg=8, bn_dim_cls=8, fusion_hidden=8):
        """
        input_dim: 输入特征维度（例如4）
        num_classes: 分类分支的类别数（非均匀划分后的类别数）
        hidden_dim: 每个分支第一层神经元个数（例如16）
        bn_dim_reg: 回归分支第二层神经元个数（例如8）
        bn_dim_cls: 分类分支第二层神经元个数（例如8）
        fusion_hidden: 融合层隐藏单元数
        """
        super(CombinedRegClsModel, self).__init__()
        # 回归分支
        self.reg_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bn_dim_reg),
            # nn.BatchNorm1d(bn_dim_reg),
            nn.ReLU(),
            nn.Linear(bn_dim_reg, 1)
        )
        # 分类分支：输出类别 logits
        self.cls_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bn_dim_cls),
            # nn.BatchNorm1d(bn_dim_cls),
            nn.ReLU(),
            nn.Linear(bn_dim_cls, num_classes)  # 输出 logits，后续交叉熵直接使用
        )

        # 融合层（移除概率的BN）
        self.fusion = nn.Sequential(
            nn.Linear(1 + num_classes, fusion_hidden),
            # nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1)
        )

        # # BatchNorm 层用于融合前调整尺度
        # self.bn_reg = nn.BatchNorm1d(bn_dim_reg)
        # self.bn_cls = nn.BatchNorm1d(num_classes)
        # # 融合层：拼接回归分支的输出（bn_dim_reg）和分类分支的概率（num_classes），再输出最终预测
        # fusion_input_dim = bn_dim_reg + num_classes
        # self.fusion = nn.Linear(fusion_input_dim, fusion_hidden)
        # self.out_layer = nn.Linear(fusion_hidden, 1)

    def forward(self, x):
        # 回归分支
        reg_feat = self.reg_branch(x)  # (batch, bn_dim_reg)

        # 分类分支
        cls_logits = self.cls_branch(x)
        cls_probs = F.softmax(cls_logits, dim=1)  # 保持原始概率分布

        # 特征融合
        combined = torch.cat([reg_feat, cls_probs], dim=1)
        y_pred = self.fusion(combined)
        return y_pred, cls_logits


class ParallelNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=16, bn_dim_reg=8, bn_dim_cls=8, fusion_hidden=8):
        super().__init__()
        # --------------------------
        # 独立回归网络（参数组1）
        # --------------------------
        self.reg_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bn_dim_reg),
            # nn.BatchNorm1d(bn_dim_reg),
            nn.ReLU(),
            nn.Linear(bn_dim_reg, 1)
        )


        # --------------------------
        # 独立分类网络（参数组2）
        # --------------------------
        self.cls_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bn_dim_cls),
            # nn.BatchNorm1d(bn_dim_cls),
            nn.ReLU(),
            nn.Linear(bn_dim_cls, num_classes)  # 输出 logits，后续交叉熵直接使用
        )

        # --------------------------
        # 融合层（参数组3）
        # --------------------------
        self.fusion_layer = nn.Sequential(
            nn.Linear(1 + num_classes, fusion_hidden),
            nn.BatchNorm1d(fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 1)
        )


    def forward(self, x):
        # 回归分支（冻结时需停止梯度）
        reg_feat = self.reg_net(x)

        # 分类分支
        cls_logits = self.cls_net(x)
        cls_probs = F.softmax(cls_logits, dim=1)

        # 融合输出
        fused = self.fusion_layer(torch.cat([reg_feat, cls_probs], dim=1))
        return fused, reg_feat, cls_logits





class MoE_Regressor(nn.Module):
    def __init__(self, num_experts, input_dim, TOP_K, DROPOUT_RATE=0.2):
        super().__init__()
        self.num_experts = num_experts
        self.TOP_K = TOP_K

        # 专家网络（含Dropout）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 16),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),  # 改进1：专家多样性增强
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),  # 改进1：专家多样性增强
                nn.Linear(8, 1)
            ) for _ in range(num_experts)
        ])

        # 门控网络 (原始嵌入实现)
        self.gate_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim)) for _ in range(num_experts)
        ])

    def forward(self, x, return_gate=False):
        # 计算门控权重 (原始公式实现)
        gate_logits = torch.stack([
            torch.matmul(x, emb) for emb in self.gate_embeddings
        ], dim=1)  # [batch_size, num_experts]

        # 改进2：Top-K稀疏门控
        topk_weights, topk_indices = torch.topk(gate_logits, k=self.TOP_K, dim=1)
        topk_weights = torch.softmax(topk_weights, dim=1)

        # 生成稀疏门控矩阵
        sparse_gate = torch.zeros_like(gate_logits).scatter(
            1, topk_indices, topk_weights
        )  # [batch_size, num_experts]

        # 计算专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, 1]

        # 加权融合
        output = torch.sum(expert_outputs * sparse_gate.unsqueeze(-1), dim=1)

        if return_gate:
            return output, sparse_gate
        return output

    def load_balance_loss(self, gate_weights):
        # 改进3：负载均衡正则项
        expert_load = gate_weights.mean(dim=0)  # [num_experts]
        load_balance_loss = torch.std(expert_load) / (torch.mean(expert_load) + 1e-8)
        return BALANCE_COEF * load_balance_loss
