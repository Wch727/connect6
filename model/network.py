"""
神经网络模型 - 基于ResNet的策略价值网络
设计原则：
1. 模型大小适中
2. 采用残差网络结构，提高训练稳定性
3. 双头输出：策略头（落子概率）和价值头（局面评估）

增强版特性：
- Bottleneck残差块（更高效）
- 多头自注意力机制（全局信息）
- 全局上下文模块（GC-Net）
- 空间金字塔池化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import math


class ConvBlock(nn.Module):
    """卷积块：Conv + BatchNorm + ReLU"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力块，提升特征表达能力"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze
        y = x.view(b, c, -1).mean(dim=2)
        # Excitation
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        y = y.view(b, c, 1, 1)
        return x * y


class SEResBlock(nn.Module):
    """带SE注意力的残差块"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


# ==================== 先进网络组件 ====================

class BottleneckBlock(nn.Module):
    """
    Bottleneck残差块 - 更高效的残差结构
    1x1 降维 -> 3x3 卷积 -> 1x1 升维
    """
    expansion = 4

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = channels // reduction

        self.conv1 = nn.Conv2d(channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out


class GlobalContextBlock(nn.Module):
    """
    全局上下文模块 (GC-Net)
    通过全局注意力池化捕获长距离依赖
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.conv_mask = nn.Conv2d(channels, 1, 1)
        self.softmax = nn.Softmax(dim=2)

        self.transform = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.LayerNorm([channels // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # 计算全局上下文
        context_mask = self.conv_mask(x)  # (b, 1, h, w)
        context_mask = context_mask.view(b, 1, h * w)
        context_mask = self.softmax(context_mask)
        context_mask = context_mask.view(b, 1, h, w)

        # 全局上下文聚合
        context = (x * context_mask).sum(dim=(2, 3), keepdim=True)  # (b, c, 1, 1)

        # 变换并添加到输入
        context = self.transform(context)
        return x + context


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # 通道注意力
        self.channel_attention = SEBlock(channels, reduction)
        # 空间注意力
        self.spatial_attention = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力模块 - 用于捕获棋盘上的全局关系
    """
    def __init__(self, channels: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # 生成Q, K, V
        qkv = self.qkv(x).reshape(b, 3, self.num_heads, self.head_dim, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (b, heads, head_dim, hw)

        # 注意力计算
        q = q.transpose(-2, -1)  # (b, heads, hw, head_dim)
        attn = (q @ k) * self.scale  # (b, heads, hw, hw)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 应用注意力
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # (b, heads, head_dim, hw)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)

        return x + out


class TransformerBlock(nn.Module):
    """
    Transformer块 - 结合自注意力和FFN
    """
    def __init__(self, channels: int, num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.attn = MultiHeadSelfAttention(channels, num_heads, dropout)
        self.norm2 = nn.GroupNorm(8, channels)

        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(mlp_hidden, channels, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AdvancedResBlock(nn.Module):
    """
    高级残差块 - 结合SE注意力、全局上下文和空间注意力
    """
    def __init__(self, channels: int, use_gc: bool = True, use_cbam: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if use_cbam:
            self.attention = CBAM(channels)
        else:
            self.attention = SEBlock(channels)

        self.gc = GlobalContextBlock(channels) if use_gc else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)
        if self.gc is not None:
            out = self.gc(out)
        out = F.relu(out + residual)
        return out


class PolicyValueNet(nn.Module):
    """
    策略价值网络 - 增强版

    输入: (batch, 17, 19, 19) 的特征平面
    输出:
        - policy: (batch, 361) 的落子概率分布
        - value: (batch, 1) 的局面评估值 [-1, 1]

    网络结构（适中大小，约2-3M参数）:
        - 输入卷积层
        - 10个残差块（带SE注意力）
        - 策略头和价值头
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 17,
        num_channels: int = 128,  # 通道数，控制模型大小
        num_res_blocks: int = 10,  # 残差块数量
        use_se: bool = True,  # 是否使用SE注意力
        dropout_rate: float = 0.1,  # Dropout率
    ):
        super().__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size
        self.dropout_rate = dropout_rate
        self.init_kwargs = {
            'board_size': board_size,
            'in_channels': in_channels,
            'num_channels': num_channels,
            'num_res_blocks': num_res_blocks,
            'use_se': use_se,
            'dropout_rate': dropout_rate,
        }

        # 输入卷积层
        self.input_conv = ConvBlock(in_channels, num_channels)

        # 残差块
        if use_se:
            self.res_blocks = nn.Sequential(*[SEResBlock(num_channels) for _ in range(num_res_blocks)])
        else:
            self.res_blocks = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])

        # 策略头 - 增强版 (优化: 32通道 -> 4通道，大幅减少全连接层参数)
        self.policy_conv = ConvBlock(num_channels, 4, kernel_size=1, padding=0)
        self.policy_bn = nn.BatchNorm1d(4 * board_size * board_size)
        self.policy_dropout = nn.Dropout(dropout_rate)
        self.policy_fc = nn.Linear(4 * board_size * board_size, self.action_size)

        # 价值头 - 增强版 (优化: 32通道 -> 4通道)
        self.value_conv = ConvBlock(num_channels, 4, kernel_size=1, padding=0)
        self.value_bn = nn.BatchNorm1d(4 * board_size * board_size)
        self.value_dropout = nn.Dropout(dropout_rate)
        self.value_fc1 = nn.Linear(4 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化网络权重（Xavier初始化）"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x: (batch, 17, 19, 19) 输入特征

        Returns:
            policy: (batch, 361) 策略logits（未经softmax）
            value: (batch, 1) 价值评估
        """
        # 共享特征提取
        x = self.input_conv(x)
        x = self.res_blocks(x)

        # 策略头
        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = self.policy_bn(p)
        p = self.policy_dropout(p)
        p = self.policy_fc(p)

        # 价值头
        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = self.value_bn(v)
        v = self.value_dropout(v)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        对单个状态进行预测

        Args:
            state: (17, 19, 19) numpy数组

        Returns:
            policy: (361,) 策略概率分布
            value: 标量价值
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)

            policy_logits, value = self(state_tensor)
            policy = F.softmax(policy_logits, dim=1)

            return policy.cpu().numpy()[0], value.cpu().item()

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量预测

        Args:
            states: (batch, 17, 19, 19) numpy数组

        Returns:
            policies: (batch, 361) 策略概率分布
            values: (batch,) 价值
        """
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            device = next(self.parameters()).device
            states_tensor = states_tensor.to(device)

            if device.type == 'cuda':
                # Use autocast for FP16 inference if on CUDA
                with torch.amp.autocast('cuda'):
                    policy_logits, values = self(states_tensor)
            else:
                policy_logits, values = self(states_tensor)

            policies = F.softmax(policy_logits, dim=1)

            return policies.cpu().numpy(), values.cpu().numpy().flatten()


class PolicyValueNetSmall(PolicyValueNet):
    """小型网络，用于快速测试"""

    def __init__(self, board_size: int = 19, in_channels: int = 17):
        super().__init__(
            board_size=board_size,
            in_channels=in_channels,
            num_channels=64,
            num_res_blocks=5,
            use_se=False,
        )
        # 修正 init_kwargs，只保留当前类接受的参数
        self.init_kwargs = {
            'board_size': board_size,
            'in_channels': in_channels,
        }


class PolicyValueNetLarge(PolicyValueNet):
    """大型网络，用于更强的性能"""

    def __init__(self, board_size: int = 19, in_channels: int = 17):
        super().__init__(
            board_size=board_size,
            in_channels=in_channels,
            num_channels=256,
            num_res_blocks=20,
            use_se=True,
        )
        # 修正 init_kwargs，只保留当前类接受的参数
        self.init_kwargs = {
            'board_size': board_size,
            'in_channels': in_channels,
        }


# ==================== 先进网络架构 ====================

class PolicyValueNetAdvanced(nn.Module):
    """
    先进策略价值网络 - 结合多种现代技术

    特性：
    1. 混合卷积残差 + Transformer架构
    2. 全局上下文模块 (GC-Net)
    3. CBAM注意力机制
    4. 多尺度特征融合
    5. 更强的策略头和价值头

    适用于需要更高性能的场景
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 17,
        num_channels: int = 192,
        num_res_blocks: int = 12,
        num_transformer_blocks: int = 2,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size
        self.init_kwargs = {
            'board_size': board_size,
            'in_channels': in_channels,
            'num_channels': num_channels,
            'num_res_blocks': num_res_blocks,
            'num_transformer_blocks': num_transformer_blocks,
            'num_heads': num_heads,
            'dropout_rate': dropout_rate,
        }

        # 输入处理 - 多尺度特征提取
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // 2, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # 主干网络：高级残差块
        self.res_blocks = nn.ModuleList([
            AdvancedResBlock(num_channels, use_gc=(i % 3 == 2), use_cbam=(i % 4 == 3))
            for i in range(num_res_blocks)
        ])

        # Transformer块 - 捕获全局依赖
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(num_channels, num_heads, mlp_ratio=2.0, dropout=dropout_rate)
            for _ in range(num_transformer_blocks)
        ])

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # 策略头 - 增强版（直接在特征图上预测）
        self.policy_conv1 = nn.Conv2d(num_channels, 64, 3, padding=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(64)
        self.policy_conv2 = nn.Conv2d(64, 2, 1, bias=False)  # 2通道：用于空间注意力
        self.policy_bn2 = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        self.policy_dropout = nn.Dropout(dropout_rate)

        # 价值头 - 增强版（全局池化 + MLP）
        self.value_conv = nn.Conv2d(num_channels, 32, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_pool = nn.AdaptiveAvgPool2d(1)
        self.value_fc1 = nn.Linear(32, 128)
        self.value_fc2 = nn.Linear(128, 64)
        self.value_fc3 = nn.Linear(64, 1)
        self.value_dropout = nn.Dropout(dropout_rate)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 输入处理
        x = self.input_conv(x)

        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)

        # Transformer块
        for trans_block in self.transformer_blocks:
            x = trans_block(x)

        # 特征融合
        x = self.fusion(x)

        # 策略头
        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = F.relu(self.policy_bn2(self.policy_conv2(p)))
        p = p.view(p.size(0), -1)
        p = self.policy_dropout(p)
        p = self.policy_fc(p)

        # 价值头
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = self.value_pool(v).view(v.size(0), -1)
        v = self.value_dropout(F.relu(self.value_fc1(v)))
        v = self.value_dropout(F.relu(self.value_fc2(v)))
        v = torch.tanh(self.value_fc3(v))

        return p, v

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                state_tensor = state_tensor.cuda()
            policy_logits, value = self(state_tensor)
            policy = F.softmax(policy_logits, dim=1)
            return policy.cpu().numpy()[0], value.cpu().item()

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            if next(self.parameters()).is_cuda:
                states_tensor = states_tensor.cuda()
                # Use autocast for FP16 inference if on CUDA
                with torch.amp.autocast('cuda'):
                    policy_logits, values = self(states_tensor)
            else:
                policy_logits, values = self(states_tensor)

            policies = F.softmax(policy_logits, dim=1)
            return policies.cpu().numpy(), values.cpu().numpy().flatten()


class PolicyValueNetHybrid(nn.Module):
    """
    混合架构网络 - 平衡性能和效率

    特性：
    1. 底层使用高效的Bottleneck块
    2. 中层使用SE注意力
    3. 顶层使用轻量Transformer
    4. 渐进式通道扩展
    """

    def __init__(
        self,
        board_size: int = 19,
        in_channels: int = 17,
        base_channels: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        self.board_size = board_size
        self.action_size = board_size * board_size
        self.init_kwargs = {
            'board_size': board_size,
            'in_channels': in_channels,
            'base_channels': base_channels,
            'dropout_rate': dropout_rate,
        }

        # 输入层
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 第一阶段：Bottleneck块 (高效特征提取)
        self.stage1 = nn.Sequential(*[
            BottleneckBlock(base_channels) for _ in range(4)
        ])

        # 第二阶段：SE残差块 (通道注意力)
        self.stage2 = nn.Sequential(*[
            SEResBlock(base_channels) for _ in range(4)
        ])

        # 第三阶段：高级残差块 (全局上下文)
        self.stage3 = nn.Sequential(*[
            AdvancedResBlock(base_channels, use_gc=True, use_cbam=False) for _ in range(2)
        ])

        # 轻量Transformer (全局关系)
        self.transformer = TransformerBlock(base_channels, num_heads=4, mlp_ratio=2.0, dropout=dropout_rate)

        # 策略头
        self.policy_head = nn.Sequential(
            nn.Conv2d(base_channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        self.policy_dropout = nn.Dropout(dropout_rate)

        # 价值头
        self.value_head = nn.Sequential(
            nn.Conv2d(base_channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.transformer(x)

        # 策略
        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_dropout(p)
        p = self.policy_fc(p)

        # 价值
        v = self.value_head(x)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)

        return p, v

    def predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                state_tensor = state_tensor.cuda()
            policy_logits, value = self(state_tensor)
            policy = F.softmax(policy_logits, dim=1)
            return policy.cpu().numpy()[0], value.cpu().item()

    def predict_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states)
            if next(self.parameters()).is_cuda:
                states_tensor = states_tensor.cuda()
                # Use autocast for FP16 inference if on CUDA
                with torch.amp.autocast('cuda'):
                    policy_logits, values = self(states_tensor)
            else:
                policy_logits, values = self(states_tensor)

            policies = F.softmax(policy_logits, dim=1)
            return policies.cpu().numpy(), values.cpu().numpy().flatten()


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_network(model_type: str = 'medium', **kwargs) -> nn.Module:
    """
    获取网络的工厂函数

    Args:
        model_type: 网络类型
            - 'small': 小型网络 (~500K参数)
            - 'medium': 中型网络 (~2M参数)
            - 'large': 大型网络 (~10M参数)
            - 'advanced': 先进架构 (~5M参数，包含Transformer)
            - 'hybrid': 混合架构 (~3M参数，平衡性能和效率)

    Returns:
        网络实例
    """
    networks = {
        'small': PolicyValueNetSmall,
        'medium': PolicyValueNet,
        'large': PolicyValueNetLarge,
        'advanced': PolicyValueNetAdvanced,
        'hybrid': PolicyValueNetHybrid,
    }

    if model_type not in networks:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(networks.keys())}")

    return networks[model_type](**kwargs)


def test_network():
    """测试网络"""
    print("测试神经网络...")
    print("=" * 60)

    # 测试不同大小的网络
    test_cases = [
        ("Small", PolicyValueNetSmall, {}),
        ("Medium", PolicyValueNet, {}),
        ("Large", PolicyValueNetLarge, {}),
        ("Advanced", PolicyValueNetAdvanced, {}),
        ("Hybrid", PolicyValueNetHybrid, {}),
    ]

    for name, NetClass, kwargs in test_cases:
        print(f"\n{name} Network:")
        print("-" * 40)

        try:
            net = NetClass(**kwargs)
            params = count_parameters(net)
            print(f"  参数量: {params:,} ({params/1e6:.2f}M)")

            # 测试前向传播
            x = torch.randn(4, 17, 19, 19)
            policy, value = net(x)
            print(f"  输入形状: {x.shape}")
            print(f"  策略输出形状: {policy.shape}")
            print(f"  价值输出形状: {value.shape}")

            # 测试单个预测
            state = np.random.randn(17, 19, 19).astype(np.float32)
            policy_np, value_np = net.predict(state)
            print(f"  单状态预测 - 策略形状: {policy_np.shape}, 价值: {value_np:.4f}")

            # 测试推理速度
            import time
            net.eval()
            x = torch.randn(32, 17, 19, 19)
            with torch.no_grad():
                start = time.time()
                for _ in range(10):
                    _ = net(x)
                elapsed = (time.time() - start) / 10
            print(f"  推理速度: {elapsed*1000:.2f}ms/batch (batch=32)")

        except Exception as e:
            print(f"  错误: {e}")

    print("\n" + "=" * 60)
    print("网络测试完成!")


if __name__ == "__main__":
    test_network()
