import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, EncoderRNN
import numpy as np 

class Model(nn.Module):
	def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
		super().__init__()

		# load graph
		self.graph = Graph(**graph_args)  # 方便地将字典中的键值对作为参数传递给函数或类构造函数
		A = np.ones((graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node']))  # (3*120*120)
# 在图中，两个节点之间的最大跳数限制为 2。加 1 的原因是要包括 0 跳的情况，即自身节点。这个矩阵用于存储从 0 跳到最大跳数（这里是 2）之间的邻接关系。在后续计算中，这个矩阵将用于构建图卷积网络的层
		# build networks
		spatial_kernel_size = np.shape(A)[0]# 3
		temporal_kernel_size = 5 #9 #5 # 3
		kernel_size = (temporal_kernel_size, spatial_kernel_size)

		# best
		# 就是Graph Convolution Model的那三层
		self.st_gcn_networks = nn.ModuleList((
			nn.BatchNorm2d(in_channels),
			Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),  # 1×1卷积扩展通道数(???代码上看是输出4 × 64*3个通道)+BN+ReLU+Conv2d+BN+(Dropout不启用) + residual为Conv2d(1×1 输出64个通道)+ReLU
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		))

		# initialize parameters for edge importance weighting
		# 相当于给每个边赋予初始权值(这个在论文哪里有体现???)
		if edge_importance_weighting: # 用于初始化ST-GCN网络的边权重，并根据edge_importance_weighting决定是否对边权重进行学习
			self.edge_importance = nn.ParameterList(
				[nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]  # 为什么要以self.st_gcn_networks为迭代对象呢?
				#将参数弄成可以进行梯度传递的形式,注册为Parameter类型,使参数可被优化器探测并优化
				)
		else:
			self.edge_importance = [1] * len(self.st_gcn_networks)

		self.num_node = num_node = self.graph.num_node #车辆总数
		self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate， 每个节点的输出为对应坐标
		self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		#或许后面几个可以不要或者换成别的车辆
		self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)  # 这3个Seq2Seq没有区别啊,那为什么要起不同的名字,不能重复使用吗
		self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)


	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() #获取对应的参数尺寸
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) #N*V此处就是合并了一些维度
		return now_feat

	def reshape_from_lstm(self, predicted):#相当于进一步改变形状
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		x = pra_x                   # pra_pred_length是预测的未来帧数
		
		# forwad
		for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
			if type(gcn) is nn.BatchNorm2d:#相当于如果是最开头的情况
				x = gcn(x)#则先把输入的坐标拿去进行batch Normalization + conv2D(1*1)    [注意]”+conv2D(1*1)”这个合理吗? gcn目前只是一个BN层而已;而且x当前在特征上是4维的, 包括了速度、方向和标记
			else:
				x, _ = gcn(x, pra_A + importance)
				
		# prepare for seq2seq lstm model
		graph_conv_feature = self.reshape_for_lstm(x)

		torch.save(graph_conv_feature, 'graph_conv_feature.pt')

		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

		if pra_teacher_forcing_ratio>0 and type(pra_teacher_location) is not type(None):
			pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

		# now_predict.shape = (N, T, V*C)，或许此处还应该加入behavior 想办法往里面加呢
		now_predict_car = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict_car = self.reshape_from_lstm(now_predict_car) # (N, C, T, V)

		#下面的不要了 或许可以改成behaivor的情况
		now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)

		now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)

		now_predict = (now_predict_car + now_predict_human + now_predict_bike)/3.  # 取3次预测的平均值

		return now_predict 

if __name__ == '__main__':
	model = Model(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
	print(model)
