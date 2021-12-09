import muspy
import glob
import pretty_midi

def name_parser(string: str, left='', right='') -> str:
	# strip off extension and path
	string = string.lstrip(left)
	string = string.rstrip(right)
	return string

def midi2json(path: str, name: str, program: muspy) -> None:
	muspy.save_json(path + name, program)	

def multi2one(program: muspy) -> muspy:
	# put all tracks into one
	tracks = program.tracks
	program.tracks = [muspy.Track(program=0, \
							is_drum=False, name='Piano')]
	# synthesize all notes into one track
	for t in tracks:
		for n in t.notes:
			program.tracks[0].notes.append(n)

