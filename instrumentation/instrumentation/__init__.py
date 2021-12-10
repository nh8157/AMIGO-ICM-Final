from . import util
import muspy
import subprocess
import os
import pretty_midi
os.chdir('accompany')
from .accompany.AccoMontage_inference import *

## accompaniment configurations
# Configurations upon inference
SPOTLIGHT = ['一曲红尘', '向天再借五百年', '夜曲', '我只在乎你', '撕夜', '放生', '明天你是否依然爱我', \
	'映山红', '浪人情歌', '海芋恋', '狂浪生', '用心良苦', '男孩', '祝我生日快乐', '背对背拥抱', '舍得', '葫芦娃', '香水', '小乌龟']
PREFILTER = None
# Song inputs
SONG_SAMPLE = {
	'Beaux of Oakhill.mid': ['A8A8B8B8\n', 0, 32],
	'Boggy Brays.mid': ['A8A8B8B8\n', 0, 32],
	'Castles in the Air.mid': ['A8A8B8B8\n', 1, 32],
	'Cuillin Reel.mid': ['A4A4B8B8\n', 1, 24],
	"Kitty O'Niel's Champion.mid": ['A4A4B4B4A4A4B4B4\n', 1, 32],
	"Proudlocks's Variation.mid": ['A8A8B8B8\n', 1, 32],
	'ECNU University Song.mid': ['A8A8B8B8C8D8E4F6A8A8B8B8C8D8E4F6\n', 0, 116]
}
SONG_ROOT = 'accompany/demo lead sheets/'

MIDI_PATH = './output/acco_output/'
INTER_PATH = './output/single_json/'
JSON_PATH = './output/arra_output/'

MODEL_PATH = './model/'

arranger_path = './arranger/arranger/lstm/predict.py'

def generate(midi_path: str, song_name: str, acco_style: list, arra_style: str, audio_path: str, uploaded_midi=False, segmentation='') -> str:
	#####################
	### accompaniment ###
	#####################
	# if the user does not upload midi file, just use the default song samples on the webpage
	if not uploaded_midi:
		os.chdir('instrumentation/instrumentation/accompany')
		processed_midi, melody_matrix, chroma_matrix, chord_table = load_and_preprocess_midi(os.path.join(midi_path, song_name), SONG_SAMPLE[song_name][1], True)
		melody_queries, piano_roll, query_phrases, query_seg = segment_phrases(melody_matrix, chroma_matrix, SONG_SAMPLE[song_name][0])
		acc_pool, texture_filter = load_and_process_reference_data()
		path, shift = select_phrase(query_phrases, melody_queries, query_seg, acc_pool, texture_filter, PREFILTER, acco_style)
		name = save_result_midi(piano_roll, chord_table, acc_pool, query_seg, path, shift)
		# could return a directory to use the file
		os.chdir('../')
	# if the user uploads customized midi file, then the user's midi is stored in ./uploads folder
	else:
		processed_midi, melody_matrix, chroma_matrix, chord_table = load_and_preprocess_midi(os.path.join(midi_path, song_name))
		melody_queries, piano_roll, query_phrases, query_seg = segment_phrases(melody_matrix, chroma_matrix, segmentation, processed_midi)
		# change the directory to complete the path-specific procedures
		os.chdir('instrumentation/instrumentation/accompany')
		acc_pool, texture_filter = load_and_process_reference_data()
		path, shift = select_phrase(query_phrases, melody_queries, query_seg, acc_pool, texture_filter, PREFILTER, acco_style)
		name = save_result_midi(piano_roll, chord_table, acc_pool, query_seg, path, shift)
		os.chdir('../')

	#######################
	### instrumentation ###
	#######################
	
	# output midi/ sound
	export_path = MIDI_PATH + name
	program = muspy.read_midi(export_path)
	name = util.name_parser(name, right='.mid') + '.json'
	# read accompaniment midi output
	# compress two-track midi into one track then convert to json
	util.multi2one(program)
	util.midi2json(INTER_PATH, name, program)
		
	# # spawn a subprocess to run arranger/lstm/predict.py
	arrangement_command = ['python3', arranger_path, '-i',  INTER_PATH+name, '-o', JSON_PATH, '-d', arra_style, '-m', MODEL_PATH + arra_style + '.hdf5', '-q']
	
	process = subprocess.Popen(arrangement_command, stdout=subprocess.PIPE)
	output, error = process.communicate()

	if error is None:
		print("Instrumentation done")
	else:
		print(error)

	# use timidity to convert the file to wav and save at $audio_path
	print("Converting midi to wav")
	name = util.name_parser(name, right='.json') + '_pred.mid'
	convert_command = ['timidity', JSON_PATH + name, '-Ow', '-o', audio_path + util.name_parser(name, right='.mid') + '.wav']
	process_conv = subprocess.Popen(convert_command, stdout=subprocess.PIPE)
	output, error = process_conv.communicate()
	if error is None:
		print("Conversion done")
	else:
		print(error)	

	name = util.name_parser(name, right='.mid') + '.wav'
	print(name, "saved at", audio_path)

	os.chdir('../..')
	return name