import numpy as np 
import glob
import os 
from scipy import spatial 
import pickle

# Please change this to your location
data_root = './Dataset/'


history_frames = 6 # 3 second * 2 frame/second
future_frames = 6 # 3 second * 2 frame/second
total_frames = history_frames + future_frames
# xy_range = 120 # max_x_range=121, max_y_range=118
max_num_object = 120 # maximum number of observed objects is 70
neighbor_distance = 10 # meter

# Baidu ApolloScape data format:
# frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading
total_feature_dimension = 10 + 1 # we add mark "1" to the end of each row to indicate that this row exists

# after zero centralize data max(x)=127.1, max(y)=106.1, thus choose 130

def get_frame_instance_dict(pra_file_path):
	'''
	Read raw data from files and return a dictionary: 
		{frame_id: 按照时刻来遍历（每个时刻的所有车辆信息形成一个字典）
			{object_id: 
				# 10 features
				[frame_id, object_id, object_type, position_x, position_y, position_z, object_length, pbject_width, pbject_height, heading]
			}
		}
	'''
	with open(pra_file_path, 'r') as reader:
		# print(train_file_path)：strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
		content = np.array([x.strip().split(' ') for x in reader.readlines()]).astype(float) # split() 通过指定分隔符对字符串进行切片
		now_dict = {}
		for row in content:
			# instance = {row[1]:row[2:]}
			n_dict = now_dict.get(row[0], {})
			n_dict[row[1]] = row#[2:]
			# n_dict.append(instance)
			# now_dict[]
			now_dict[row[0]] = n_dict
	return now_dict

def process_data(pra_now_dict, pra_start_ind, pra_end_ind, pra_observed_last):
	visible_object_id_list = list(pra_now_dict[pra_observed_last].keys()) # object_id appears at the last observed frame  #只是获得ID
	num_visible_object = len(visible_object_id_list) # number of current observed objects  个人认为在history_frames期间的最后一帧仍然存在的object才是可观察object

	# compute the mean values of x and y for zero-centralization. 
	visible_object_value = np.array(list(pra_now_dict[pra_observed_last].values()))#此处是获得相关的feature（10 features）
	xy = visible_object_value[:, 3:5].astype(float)#取得对应各个节点的坐标（x,y),相当于一整个列表
	mean_xy = np.zeros_like(visible_object_value[0], dtype=float)#初始化为汽车总数（前一时刻观察到的)
	m_xy = np.mean(xy, axis=0)
	mean_xy[3:5] = m_xy

	# compute distance between any pair of two objects
	dist_xy = spatial.distance.cdist(xy, xy)#计算两两节点之间的相互距离
	# if their distance is less than $neighbor_distance, we regard them are neighbors.
	neighbor_matrix = np.zeros((max_num_object, max_num_object))
	neighbor_matrix[:num_visible_object, :num_visible_object] = (dist_xy<neighbor_distance).astype(int)#创建邻接矩阵（根据上述的距离矩阵来创建，距离小于设定阈值则标记为1）
	#pra_now_dict[frame_ind][obj_id]
	now_all_object_id = set([val for x in range(pra_start_ind, pra_end_ind) for val in pra_now_dict[x].keys()])#pra_now_dict是一个按照时刻分的字典，相当于get_frame_instance_dict中的函数，此时取出的应该是object_id
	non_visible_object_id_list = list(now_all_object_id - set(visible_object_id_list))#拿现在的Id - 之前的Id = 最新时刻新增ID数量
	num_non_visible_object = len(non_visible_object_id_list)#得到最新时刻新增的ID数量
	
	# for all history frames(6) or future frames(6), we only choose the objects listed in visible_object_id_list
	object_feature_list = []
	# non_visible_object_feature_list = []
	for frame_ind in range(pra_start_ind, pra_end_ind):	
		# we add mark "1" to the end of each row to indicate that this row exists, using list(pra_now_dict[frame_ind][obj_id])+[1] 
		# -mean_xy is used to zero_centralize data
		# now_frame_feature_dict = {obj_id : list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] for obj_id in pra_now_dict[frame_ind] if obj_id in visible_object_id_list}
		now_frame_feature_dict = {obj_id : (list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[1] if obj_id in visible_object_id_list else list(pra_now_dict[frame_ind][obj_id]-mean_xy)+[0]) for obj_id in pra_now_dict[frame_ind] }
		# if the current object is not at this frame, we return all 0s by using dict.get(_, np.zeros(11))
		now_frame_feature = np.array([now_frame_feature_dict.get(vis_id, np.zeros(total_feature_dimension)) for vis_id in visible_object_id_list+non_visible_object_id_list])
		object_feature_list.append(now_frame_feature)   # now_frame_feature包含某个frame的object's feature, shape为12帧总object数×total_feature_dimension

	# object_feature_list has shape of (frame#, object#, 11) 11 = 10features + 1mark
	object_feature_list = np.array(object_feature_list) # shape为帧数×12帧总object数×特征维度
	
	# object feature with a shape of (frame#, object#, 11) -> (object#, frame#, 11)
	object_frame_feature = np.zeros((max_num_object, pra_end_ind-pra_start_ind, total_feature_dimension))

	# np.transpose(object_feature_list, (1,0,2))
	object_frame_feature[:num_visible_object+num_non_visible_object] = np.transpose(object_feature_list, (1,0,2))   # 这个操作相当于把特征按object来划分,一共28个object,故前28个二维数组是有值的,每个二维数组的行按total_frames排列
	# object_frame_feature 形状为 最大物体数×total_frames×total_feature_dimension
	return object_frame_feature, neighbor_matrix, m_xy
	

def generate_train_data(pra_file_path):
	'''
	Read data from $pra_file_path, and split data into clips with $total_frames length. 
	Return: feature and adjacency_matrix
		feture: (N, C, T, V) 
			N is the number of training data 
			C is the dimension of features, 10raw_feature + 1mark(valid data or not)
			T is the temporal length of the data. history_frames + future_frames2
			V is the maximum number of objects. zero-padding for less objects. 
			
	'''
	now_dict = get_frame_instance_dict(pra_file_path)
	frame_id_set = sorted(set(now_dict.keys()))

	all_feature_list = []
	all_adjacency_list = []
	all_mean_list = []
	for start_ind in frame_id_set[:-total_frames+1]: # 有点像一个长为total_frames的块在整个txt文件包含的所有frames中逐格推进
		start_ind = int(start_ind)     # frame_id_set[:-total_frames+1]舍弃倒数的11个元素(-total_frames+1)
		end_ind = int(start_ind + total_frames)
		observed_last = start_ind + history_frames - 1
		object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

		all_feature_list.append(object_frame_feature) # 添加一个total_frames期间内的物体特征, 初始的”维度”代表有多少个total_frames片段, 称为维度有点不恰当, 因为all_feature_list还是个list
		all_adjacency_list.append(neighbor_matrix)	
		all_mean_list.append(mean_xy)	

	# (Number of total_frames, max_num_object, total_frames, total_feature_dimension) --> (Number of total_frames, total_feature_dimension, total_frames, max_num_object)
	all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
	all_adjacency_list = np.array(all_adjacency_list)  # all_adjacency_list由所有total_frames片段的邻接矩阵构成, shape为total_frames片段数×max_num_object×max_num_object
	all_mean_list = np.array(all_mean_list)  # all_mean_list存放着所有total_frames片段中由visible区域的物体xy坐标均值构成的参考系  Number of total_frames×2
	# print(all_feature_list.shape, all_adjacency_list.shape)
	return all_feature_list, all_adjacency_list, all_mean_list


def generate_test_data(pra_file_path):
	now_dict = get_frame_instance_dict(pra_file_path)
	frame_id_set = sorted(set(now_dict.keys()))
	
	all_feature_list = []
	all_adjacency_list = []
	all_mean_list = []
	# get all start frame id
	start_frame_id_list = frame_id_set[::history_frames]  # 这种切片操作尚未查阅
	for start_ind in start_frame_id_list:              # test中每隔history_frames所选中的片段是连续的, 相当于是prediction_test.txt中没有给future_frames
		start_ind = int(start_ind)
		end_ind = int(start_ind + history_frames)
		observed_last = start_ind + history_frames - 1
		# print(start_ind, end_ind)
		object_frame_feature, neighbor_matrix, mean_xy = process_data(now_dict, start_ind, end_ind, observed_last)

		all_feature_list.append(object_frame_feature)
		all_adjacency_list.append(neighbor_matrix)	
		all_mean_list.append(mean_xy)

	# (N, V, T, C) --> (N, C, T, V)
	all_feature_list = np.transpose(all_feature_list, (0, 3, 2, 1))
	all_adjacency_list = np.array(all_adjacency_list)
	all_mean_list = np.array(all_mean_list)
	# print(all_feature_list.shape, all_adjacency_list.shape)
	return all_feature_list, all_adjacency_list, all_mean_list


def generate_data(pra_file_path_list, pra_is_train=True):
	all_data = []
	all_adjacency = []
	all_mean_xy = []
	for file_path in pra_file_path_list:
		if pra_is_train:
			now_data, now_adjacency, now_mean_xy = generate_train_data(file_path)
		else:
			now_data, now_adjacency, now_mean_xy = generate_test_data(file_path)
		all_data.extend(now_data)    # 这3个都是按照total_frames片段堆叠起来的
		all_adjacency.extend(now_adjacency)
		all_mean_xy.extend(now_mean_xy)

	all_data = np.array(all_data) #(N, C, T, V)=(5010, 11, 12, 70) Train
	all_adjacency = np.array(all_adjacency) #(5010, 70, 70) Train
	all_mean_xy = np.array(all_mean_xy) #(5010, 2) Train

	# Train (N, C, T, V)=(5010, 11, 12, 70), (5010, 70, 70), (5010, 2)
	# Test (N, C, T, V)=(415, 11, 6, 70), (415, 70, 70), (415, 2)
	print(np.shape(all_data), np.shape(all_adjacency), np.shape(all_mean_xy))

	# save training_data and training_adjacency into a file.
	if pra_is_train:
		save_path = 'train_data.pkl'
	else:
		save_path = 'test_data.pkl'
	with open(save_path, 'wb') as writer:
		pickle.dump([all_data, all_adjacency, all_mean_xy], writer)


if __name__ == '__main__':
	train_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_train/*.txt')))
	test_file_path_list = sorted(glob.glob(os.path.join(data_root, 'prediction_test/*.txt')))

	print('Generating Training Data.')
	generate_data(train_file_path_list, pra_is_train=True)
	
	print('Generating Testing Data.')
	generate_data(test_file_path_list, pra_is_train=False)
	


