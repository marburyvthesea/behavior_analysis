import os
import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/johnmarshall/Documents/Analysis/PythonAnalysisScripts/post_cmfe_analysis')
import dlc_utils 
from importlib import reload
import matplotlib.pyplot as plt




def create_velocity_df(df_with_centroid):
	difference_df = dlc_utils.difference_df(df_with_centroid)
	df_columns = difference_df.columns

	velocity_df = pd.DataFrame(np.transpose(np.array([np.array([dlc_utils.velocity(difference_df[body_part]['x'].values[frame],difference_df[body_part]['y'].values[frame]) 
			for frame in range(len(difference_df))]) for body_part in list(set([df_columns[item][0] 
			for item in range(len(df_columns))]))])), columns=list(set([df_columns[item][0] for item in range(len(df_columns))]))) 

	difference_df_time_delta = difference_df.set_index(pd.to_timedelta(np.linspace(0, len(difference_df)*(1/4), len(difference_df)), unit='s'), drop=False)
	#velocity_df = dlc_utils.velocity_df_from_difference_df(difference_df)
	#need to reset velocity df index to time delta 
	velocity_df = velocity_df.set_index(pd.to_timedelta(np.linspace(0, len(velocity_df)*(1/4), len(velocity_df)), unit='s'), drop=False)

	df_with_centroid = df_with_centroid.set_index(velocity_df.index)

	# downsample and interpolate to find pauses between movements
	interpolated = dlc_utils.downsample_and_interpolate(velocity_df, '.2S', '1S', 'linear')

	##want to get the regions where the mouse is stopped for some time threshold
	#use bin by activity threhold function

	resting_bins = dlc_utils.bin_by_resting_threshold(interpolated['centroid'], 5, 5, 5, 4, 7)

	resting_bin_indicies = dlc_utils.select_trigger_regions(resting_bins, 0.5, 0.5, 5)
	resting_bins_time_delta = pd.DataFrame(resting_bins).set_index(interpolated.index, drop=False)

	return(resting_bin_indicies, resting_bins_time_delta, df_with_centroid, difference_df_time_delta)


def calculate_tortuosity(resting_bin_indicies, resting_bins_time_delta, df_with_centroid, difference_df_time_delta):

	sl_distances = np.zeros(len(resting_bin_indicies))
	path_distances = np.zeros(len(resting_bin_indicies))
	time_start = []

	for resting_idx in np.linspace(0, len(resting_bin_indicies)-3, len(resting_bin_indicies)-2):
		start_index = df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].index.get_loc(resting_bins_time_delta.index[resting_bin_indicies[int(resting_idx)]], method='nearest')
		end_index = df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].index.get_loc(resting_bins_time_delta.index[resting_bin_indicies[int(resting_idx)+1]], method='nearest')
		sl_distance = np.linalg.norm(np.array(df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[start_index]['x'], df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[start_index]['y'])-
			np.array(df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[end_index]['x'], df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[end_index]['y']))
	#print(sl_distance)
		sl_distances[int(resting_idx)] = sl_distance
		point_distances = np.zeros(end_index-start_index)
		for point, list_idx in zip(np.linspace(start_index, end_index-1, end_index-start_index), np.linspace(0, end_index-start_index-1, end_index-start_index)):
			list_idx=int(list_idx)
			#could have a "noise filter" here to control for back and forth movement in same place 
			point_distances[list_idx] = np.linalg.norm(np.array(df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[int(point)]['x'], df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[int(point)]['y'])-
				np.array(df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[int(point+1)]['x'], df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].iloc[int(point+1)]['y']))
		path_distance = np.sum(point_distances)
		#print(path_distance)
		path_distances[int(resting_idx)] = path_distance
		#print(df_with_centroid.index[resting_idx])
		#print(resting_idx)
		#time_start[int(resting_idx)] = df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].index[start_index]
		time_start.append(df_with_centroid['DLC_resnet50_mglur5ko_openfieldJun10shuffle1_300000']['centroid'].index[start_index])
		#time_start[int(resting_idx)] = df_with_centroid.index[resting_idx]
	time_start.extend(np.zeros(int(len(path_distances)-len(time_start))))
	output_df = pd.DataFrame({'straight_line_distances': sl_distances, 'path_distances': path_distances, 'tortuosity': path_distances/sl_distances, 'video_time': time_start})
	return(output_df)



