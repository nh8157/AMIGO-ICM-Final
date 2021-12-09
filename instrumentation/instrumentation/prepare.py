import util
import glob
import muspy

def process_midi(input_path, multi_output_path, single_output_path):	
	# read list of midi files
	multi_list = glob.glob(input_path + '*')
	
	for midi in multi_list:
		try:
			file_name = util.name_parser(midi, input_path, '.MID') + '.json'
			# open individual midi file
			program = muspy.read_midi(midi)
			# save a copy of multi track file
			util.midi2json(multi_output_path, file_name, program)
			# combine multi track into one track
			util.multi2one(program)
			# save a copy of single track file
			util.midi2json(single_output_path, file_name, program)
			print("Outputting", file_name, end='r')
		except:
			print("Cannot open", midi)
if __name__ == '__main__':
	input_path = './RWC/midi_multi_track/'
	multi_output_path = './RWC/json_multi_track/'
	single_output_path = './RWC/json_single_track/'
	process_midi(input_path, multi_output_path, single_output_path)
	print("Processing completed")
