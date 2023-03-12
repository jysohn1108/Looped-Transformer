import argparse

def parse_arguments():
	parser = argparse.ArgumentParser(description='Looped TF')
	parser.add_argument('--mode', type=str, default='subleq', help='name of program to run')
	parser.add_argument('--save_all', type=int, default=0, help='flag for saving all csv files')
	parser.add_argument('--output_folder', type=str, default='outputs', help='name of the output folder')
	parser.add_argument('--heatmap_folder', type=str, default='attention_heatmap', help='name of the folder saving the heatmap')
	parser.add_argument('--num_loops', type=int, default=1, help='number of loops')
	parser.add_argument('--num_trial', type=int, default=1, help='number of random trials')
	parser.add_argument('--lam', type=float, default=10, help='lambda used for softmax')

	parser.add_argument('--subleq_smnN', help='list [s,m,n,N] for subleq', type=str)
	# parser.add_argument('--small_size', type=int, default=1, help='flag for using small size')
	
	opt = parser.parse_args()

	return opt


def run_args():
	global opt
	opt = parse_arguments()
	print(opt)

run_args()