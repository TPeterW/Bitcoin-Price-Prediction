import os.path

# manage descriptive name here...
def input_file_to_output_name(filename):
	get_base_file = os.path.basename(filename)
	base_filename = get_base_file.split('.')[0]
	# base_filename = '/pipeline_data/' + base_filename
	return base_filename