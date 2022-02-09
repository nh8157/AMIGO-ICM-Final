
class Keyboard:
	def __init__(self, signature):
		self.tone_dict = {
			'major': [2, 2, 1, 2, 2, 2, 1],
			'minor': [2, 1, 2, 2, 1, 3, 1], # using harmonic minor scale
		}
		self.signature = signature
		#  C     D     E  F     G     A     B
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		self.keyboard = self.get_keyboard()

	def get_keyboard(self):
		scale = self.signature.split('/')
		root, tonality, key_sig = scale[0], scale[1], scale[2]
		key_number = self.key2number(root, tonality)
		#			C     D     E  F     G     A     B
		keyboard = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		pattern = self.tone_dict.get(key_sig)
		if key_sig == None:
			print("Key signature must be major or minor")
		else:
			for step in pattern:
				keyboard[key_number] ^= 1
				key_number = (key_number + step) % 12 
		# print(keyboard)
		return keyboard

	def key2number(self, key: str, tonality: str) -> int:
		num = (ord(key) - ord('C') + 7) % 7
		key_number = num * 2 if num <= 2 else num * 2 - 1
		if tonality == 's':
			key_number += 1
		elif tonality == 'f':
			key_number -= 1
		return key_number

	def get_chord(self, key: str) -> list:
		# expected input: [key]/[tonality]	
		try:
			key = key.split('/')
			key_number = self.key2number(key[0], key[1])
			chord = [key_number]
			# first check if the key is actually on the keyboard
			if not self.keyboard[key_number]:
				print("Not a valid key")
			else:
				if self.keyboard[(key_number + 3) % 12]:
					chord.append((key_number + 3) % 12)
				else:
					chord.append((key_number + 4) % 12)
				if self.keyboard[(key_number + 7) % 12]:
					chord.append((key_number + 7) % 12)
				else:
					chord.append((key_number + 8) % 12)
			return chord
		except:
			print("invalid input")
			return None

if __name__ == '__main__':
	kb = Keyboard('D/n/major')
	while True:
		key = input('Type in key: ')
		print(kb.get_chord(key))
