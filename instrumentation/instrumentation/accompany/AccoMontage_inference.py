from muspy import music
from numpy.core.fromnumeric import sort
from numpy.testing._private.utils import print_assert_equal
import pretty_midi as pyd 
import numpy as np 
from tqdm import tqdm
import pandas as pd 
import torch
import random
import time
import datetime
import music21
from .chord_generate_helper import Keyboard

import sys

sys.path.append('./util_tools')
from acc_utils import split_phrases, melodySplit, chordSplit, computeTIV, chord_shift, cosine, cosine_rhy, accomapnimentGeneration
import format_converter as cvt

sys.path.append('./models')
from model import DisentangleVAE

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device('cpu')

def find_by_length(melody_data, acc_data, chord_data, length):
    melody_record = []
    acc_record = []
    chord_record = []
    song_reference = []
    for song_idx in tqdm(range(acc_data.shape[0])):
        for phrase_idx in range(len(acc_data[song_idx])):
            melody = melody_data[song_idx][phrase_idx]
            if not melody.shape[0] == length * 16:
                continue
            if np.sum(melody[:, :128]) <= 2:
                continue
            melody_record.append(melody)
            acc = acc_data[song_idx][phrase_idx]
            acc_record.append(acc)
            chord = chord_data[song_idx][phrase_idx]
            chord_record.append(chord)
            song_reference.append((song_idx, phrase_idx))
    return np.array(melody_record), np.array(acc_record), np.array(chord_record), song_reference

def new_new_search(query_phrases, seg_query, acc_pool, edge_weights, texture_filter=None, filter_id=None, spotlights=None):
    print('Searching for Phrase 1')
    query_length = [query_phrases[i].shape[0]//16 for i in range(len(query_phrases))]
    mel, acc, chord, song_ref = acc_pool[query_length[0]]
    mel_set = mel
    rhy_set = np.concatenate((np.sum(mel_set[:, :, :128], axis=-1, keepdims=True), mel_set[:, :, 128: 130]), axis=-1)
    query_rhy = np.concatenate((np.sum(query_phrases[0][:, : 128], axis=-1, keepdims=True), query_phrases[0][:, 128: 130]), axis=-1)[np.newaxis, :, :]
    rhythm_result = cosine_rhy(query_rhy, rhy_set)

    chord_set = chord
    chord_set, num_total, shift_const = chord_shift(chord_set)
    chord_set_TIV = computeTIV(chord_set)
    query_chord = query_phrases[0][:, 130:][::4]
    query_chord_TIV = computeTIV(query_chord)[np.newaxis, :, :]
    chord_score, arg_chord = cosine(query_chord_TIV, chord_set_TIV)
    score = .5*rhythm_result + .5*chord_score
    if not spotlights == None:
        for spot_idx in spotlights:
            for ref_idx, ref_item in enumerate(song_ref):
                    if ref_item[0] == spot_idx: 
                        score[ref_idx] += 1

    if not filter_id == None:
        mask = texture_filter[query_length[0]][0][filter_id[0]] * texture_filter[query_length[0]][1][filter_id[1]] - 1
        score += mask

    #print(np.argmax(score), np.max(score), score[0])
    path = [[(i, score[i])] for i in range(acc.shape[0])]
    shift = [[shift_const[i]] for i in arg_chord]
    melody_record = np.argmax(mel_set, axis=-1)
    record = []
    for i in range(1, len(query_length)):
        print('Searching for Phrase', i+1)
        mel, acc, chord, song_ref = acc_pool[query_length[i]]

        weight_key = 'l' + str(query_length[i-1]) + str(query_length[i])
        contras_result = edge_weights[weight_key]
        #contras_result = (contras_result - 0.9) * 10   #rescale contrastive result if necessary
        #print(np.sort(contras_result[np.random.randint(2000)][-20:]))
        if query_length[i-1] == query_length[i]:
            for j in range(contras_result.shape[0]):
                contras_result[j, j] = -1   #the ith phrase does not transition to itself at i+1
                for k in range(j-1, -1, -1):
                    if song_ref[k][0] != song_ref[j][0]:
                        break
                    contras_result[j, k] = -1   #ith phrase does not transition to its ancestors in the same song.
        #contras_result = (contras_result - 0.99) * 100
        if i > 1:
            contras_result = contras_result[[item[-1][1] for item in record]]

        if not spotlights == None:
            for spot_idx in spotlights:
                for ref_idx, ref_item in enumerate(song_ref):
                    if ref_item[0] == spot_idx:
                        contras_result[:, ref_idx] += 1

        mel_set = mel
        rhy_set = np.concatenate((np.sum(mel_set[:, :, :128], axis=-1, keepdims=True), mel_set[:, :, 128: 130]), axis=-1)
        query_rhy = np.concatenate((np.sum(query_phrases[i][:, : 128], axis=-1, keepdims=True), query_phrases[i][:, 128: 130]), axis=-1)[np.newaxis, :, :]
        rhythm_result = cosine_rhy(query_rhy, rhy_set)
        chord_set = chord
        chord_set, num_total, shift_const = chord_shift(chord_set)
        chord_set_TIV = computeTIV(chord_set)
        query_chord = query_phrases[i][:, 130:][::4]
        query_chord_TIV = computeTIV(query_chord)[np.newaxis, :, :]
        chord_score, arg_chord = cosine(query_chord_TIV, chord_set_TIV)
        sim_this_layer = .5*rhythm_result + .5*chord_score
        if not spotlights == None:
            for spot_idx in spotlights:
                for ref_idx, ref_item in enumerate(song_ref):
                    if ref_item[0] == spot_idx: 
                        sim_this_layer[ref_idx] += 1
        score_this_layer = .7*contras_result +  .3*np.tile(sim_this_layer[np.newaxis, :], (contras_result.shape[0], 1)) + np.tile(score[:, np.newaxis], (1, contras_result.shape[1]))
        melody_flat =  np.argmax(mel_set, axis=-1)
        if seg_query[i] == seg_query[i-1]:
            melody_pre = melody_record
            matrix = np.matmul(melody_pre, np.transpose(melody_flat, (1, 0))) / (np.linalg.norm(melody_pre, axis=-1)[:, np.newaxis]*(np.linalg.norm(np.transpose(melody_flat, (1, 0)), axis=0))[np.newaxis, :])
            if i == 1:
                for k in range(matrix.shape[1]):
                    matrix[k, :k] = -1
            else:
                for k in range(len(record)):
                    matrix[k, :record[k][-1][1]] = -1
            matrix = (matrix > 0.99) * 1.
            #print(matrix.any() == 1)
            #print(matrix.shape)
            score_this_layer += matrix
        #print(score_this_layer.shape)
        #print('score_this_layer:', score_this_layer.shape)
        topk = 1
        args = np.argsort(score_this_layer, axis=0)[::-1, :][:topk, :]
        #print(args.shape, 'args:', args[:10, :2])
        #argmax = args[0, :]
        record = []
        for j in range(args.shape[-1]):
            for k in range(args.shape[0]):
                record.append((score_this_layer[args[k, j], j], (args[k, j], j)))
        
        shift_this_layer = [[shift_const[k]] for k in arg_chord]

        new_path = [path[item[-1][0]] + [(item[-1][1], sim_this_layer[item[-1][1]])] for item in record]
        new_shift = [shift[item[-1][0]] + shift_this_layer[item[-1][1]] for item in record]

        melody_record = melody_flat[[item[-1][1] for item in record]]
        path = new_path
        shift = new_shift
        score = np.array([item[0] for item in record])

    arg = score.argsort()[::-1]
    return [path[arg[i]] for i in range(topk)], [shift[arg[i]] for i in range(topk)]

def render_acc(pianoRoll, chord_table, query_seg, indices, shifts, acc_pool):
    acc_emsemble = np.empty((0, 128))
    for i, idx in enumerate(indices):
        length = int(query_seg[i][1:])
        shift = shifts[i]
        acc_matrix = np.roll(acc_pool[length][1][idx[0]], shift, axis=-1)
        acc_emsemble = np.concatenate((acc_emsemble, acc_matrix), axis=0)
    #print(acc_emsemble.shape)
    acc_emsemble = melodySplit(acc_emsemble, WINDOWSIZE=32, HOPSIZE=32, VECTORSIZE=128)
    chord_table = chordSplit(chord_table, 8, 8)
    #print(acc_emsemble.shape, chord_table.shape)
    pianoRoll = melodySplit(pianoRoll, WINDOWSIZE=32, HOPSIZE=32, VECTORSIZE=142)

    model = DisentangleVAE.init_model(device)
    checkpoint = torch.load('./data files/model_master_final.pt', map_location='cpu')
    model.load_state_dict(checkpoint)
    pr_matrix = torch.from_numpy(acc_emsemble).float().to(device)
    #pr_matrix_shifted = torch.from_numpy(pr_matrix_shifted).float().cuda()
    gt_chord = torch.from_numpy(chord_table).float().to(device)
    #print(gt_chord.shape, pr_matrix.shape)
    est_x = model.inference(pr_matrix, gt_chord, sample=False)
    #print('est:', est_x.shape)
    #est_x_shifted = model.inference(pr_matrix_shifted, gt_chord, sample=False)
    midiReGen = accomapnimentGeneration(pianoRoll, est_x, 120)
    return midiReGen
    #midiReGen.write('accompaniment_test_NEW.mid')

def ref_spotlight(ref_name_list):
    df = pd.read_excel("./data files/POP909 4bin quntization/four_beat_song_index.xlsx")
    check_idx = []
    for name in ref_name_list:
        line = df[df.name == name]
        if not line.empty:
            check_idx.append(line.index)#read by pd, neglect first row, index starts from 0.
    #print(check_idx)
    for name in ref_name_list:
        line = df[df.artist == name]
        if not line.empty:
            check_idx += list(line.index)#read by pd, neglect first row, index starts from 0
    return check_idx

def gen_key_sig(base_note_pitch):
    # given a pitch number, return its chroma format for the input of Keyboard().get_chord()
    base_note_name = pyd.note_number_to_name(base_note_pitch)
    # it has sharp
    if len(base_note_name) == 3:
        return base_note_name[0] + '/s'
    # it is natural
    elif len(base_note_name) == 2:
        return base_note_name[0] + '/n'
    else:
        print("not valid key sig!")
        return 'C/n'

def get_onset_melody_note_every_few_bar(keyboard_lst, sorted_melody_notes, downbeat, step_size):
    # for every step_size (>= 1) bars, get the first appearing melody note
    i = j = 0
    onset_note_lst = [None for n in range(0, len(downbeat), step_size)]
    while i < len(sorted_melody_notes) and j < len(downbeat):
        current = sorted_melody_notes[i]
        if current.start >= downbeat[j]:
            # only when this note is a valid note on the current key signature, we will add the note as a base note for chord generation
            current_relative_pitch = current.pitch % 12
            if keyboard_lst[current_relative_pitch] == 1:
                if current.pitch - 12 >= 21:
                    current.pitch = current.pitch - 12
                onset_note_lst[j//step_size] = current
                j += step_size
        i += 1
    # loop through the result list and see whether there are some missing parts, fill it with previous values
    for k in range(len(onset_note_lst)):
        if onset_note_lst[k] is None:
            if k == 0:
                onset_note_lst[k] = sorted_melody_notes[0]
            else:
                onset_note_lst[k] = onset_note_lst[k-1]
    return onset_note_lst

def load_and_preprocess_midi(midi_path, note_shift=0, is_default_music=False):
    midi = pyd.PrettyMIDI(midi_path)
    # if the input is (more than) two tracks of piano, read two tracks
    if len(midi.instruments) > 1:
        melody_track, chord_track = midi.instruments[0], midi.instruments[1]
    # if only one track, generate chord based on the melody track
    elif len(midi.instruments) == 1:
        melody_track = midi.instruments[0]
        chord_track = []
    else:
        print("No valid instruments and track in Midi file!")
        return None, None

    # if there are further processing of current midi, save it here
    processed_midi = pyd.PrettyMIDI()

    # get the downbeats of the midi_file
    downbeats = midi.get_downbeats()
    print("downbeat", downbeats)

    # here check whether the number of bars of the melody is odd or even
    if not is_default_music:
        number_of_bars = len(downbeats)
        if number_of_bars % 2 == 1:
            print("odd number of bars, need further pre-processing")
            # if odd number of bars, extend the music length by 1 bar
            one_bar_length_sample = downbeats[1] - downbeats[0]
            final_note = sorted(melody_track.notes, key=lambda x: x.end, reverse=True)[0]
            extended_end_time = final_note.end + one_bar_length_sample
            extended_final_note = pyd.Note(velocity=final_note.velocity, pitch=final_note.pitch, start=final_note.end, end=extended_end_time)
            melody_track.notes.append(extended_final_note)
            if chord_track:
                # if the chord track is valid, also check the bar length and do the possible extension
                final_chord_note = sorted(chord_track.notes, key=lambda x: x.end, reverse=True)[0]
                extended_final_chord_note = pyd.Note(velocity=final_chord_note.velocity, pitch=final_chord_note.pitch, start=final_chord_note.end, end=extended_end_time)
                chord_track.notes.append(extended_final_chord_note)
    downbeats = midi.get_downbeats()
    print("current downbeat", downbeats)

    # the matrix for melody track
    melody_matrix = cvt.melody_data2matrix(melody_track, downbeats)# T*130, quantized at 16th note
    if not note_shift == 0:
        melody_matrix = np.concatenate((melody_matrix[int(note_shift*4):, :], melody_matrix[-int(note_shift*4):, :]), axis=0)
    # the matrix for chord track
    try:
        # if there is no chord track, or the second instrument track does not satisfy the 3 or 4 chroma tones at a time, will generate new chords
        chroma = cvt.chord_data2matrix(chord_track, downbeats, 'quarter')  # T*36, quantized at 16th note (quarter beat)
    except:
        print("The input accompaniment chord has error, auto generating a new one for you...")
        # get the key signature of this midi file (need to input midi file name)
        score = music21.converter.parse(midi_path)
        song_key_obj = score.analyze('key')
        if len(str(song_key_obj.tonic.name)) == 1:
            song_key = str(song_key_obj.tonic.name) + '/n/' + str(song_key_obj.mode)
        elif len(str(song_key_obj.tonic.name)) == 2:
            song_key = str(song_key_obj.tonic.name)[0] + '/f/' + str(song_key_obj.mode)
        print("The analyzed song key is", song_key)
        key_keyboard = Keyboard(song_key)

        sorted_melody_notes = sorted(melody_track.notes, key=lambda x: x.start)
        melody_start_time = sorted_melody_notes[0].start
        melody_end_time = melody_track.get_end_time()
        print(melody_start_time, melody_end_time)
        # we will generate a new "chord" chorma every `bar_step_size` bars
        bar_step_size = 4
        onset_note_list = get_onset_melody_note_every_few_bar(key_keyboard.get_keyboard(), sorted_melody_notes, downbeats, bar_step_size)
        print("the selected onset notes", onset_note_list)
        chord_track = pyd.Instrument(0)
        for i in range(0, len(downbeats), bar_step_size):
            if i == 0 and downbeats[i] < melody_start_time:
                start_time = melody_start_time
            else:
                start_time = downbeats[i]
            if i + bar_step_size >= len(downbeats):
                end_time = melody_end_time
            else:
                end_time = downbeats[i+bar_step_size]
            onset_note = onset_note_list[i//bar_step_size]
            onset_note_octave = onset_note.pitch // 12
            chord_key = gen_key_sig(onset_note.pitch)
            onset_chord_list = key_keyboard.get_chord(chord_key)
            
            for offset in onset_chord_list:
                note = pyd.Note(velocity=onset_note.velocity, pitch=offset + onset_note_octave*12, start=start_time, end=end_time)
                chord_track.notes.append(note)
            print("chord notes", chord_track.notes)
        chroma = cvt.chord_data2matrix(chord_track, downbeats, 'quarter')

    if not note_shift == 0:
        chroma = np.concatenate((chroma[int(note_shift*4):, :], chroma[-int(note_shift*4):, :]), axis=0)
    chord_table = chroma[::4, :] #T'*36, quantized at 4th notes
    # chord_table[-8:, :] = chord_table[56:64, :]
    chroma = chroma[:, 12: -12] #T*12, quantized at 16th notes

    processed_midi.instruments.append(melody_track)
    processed_midi.instruments.append(chord_track)
    return processed_midi, melody_matrix, chroma, chord_table

def segment_phrases(melody_matrix, chroma, segmentation='', processed_midi=None):
    piano_roll = np.concatenate((melody_matrix, chroma), axis=-1)    #T*142, quantized at 16th

    # if segmentation is not specified, defaultly generate a segmentation
    if not segmentation:
        downbeats = processed_midi.get_downbeats()
        number_of_segments = len(downbeats)
        if number_of_segments % 2 or number_of_segments <= 2:
            print(number_of_segments)
            print("cannot parse the segments!")
            return None
        else:
            segmentation = ''
            if number_of_segments % 4 == 0:
                for i in range(number_of_segments//4):
                    segmentation += 'A4'
            else:
                for i in range(number_of_segments//4-1):
                    segmentation += 'A4'
                segmentation += 'B6'
        print("No segmentation specified, auto generated:", segmentation)

    segmentation = segmentation.strip() + '\n'
    query_phrases = split_phrases(segmentation) #[('A', 8, 0), ('A', 8, 8), ('B', 8, 16), ('B', 8, 24)]
    query_seg = [item[0] + str(item[1]) for item in query_phrases]  #['A8', 'A8', 'B8', 'B8']
    
    melody_queries = []
    for item in query_phrases:
        start_bar = item[-1]
        length = item[-2]
        segment = piano_roll[start_bar*16: (start_bar+length)*16]
        melody_queries.append(segment)  #melody queries: list of T16*142, segmented by phrases
    return melody_queries, piano_roll, query_phrases, query_seg

def load_and_process_reference_data():
    print('Loading Reference Data')
    print(os.getcwd())
    data = np.load('./data files/phrase_data0714.npz', allow_pickle=True)
    melody = data['melody']
    acc = data['acc']
    chord = data['chord']

    print('Processing Reference Phrases')
    acc_pool = {}
    (mel, acc_, chord_, song_reference) = find_by_length(melody, acc, chord, 8)
    acc_pool[8] = (mel, acc_, chord_, song_reference)

    (mel, acc_, chord_, song_reference) = find_by_length(melody, acc, chord, 4)
    acc_pool[4] = (mel, acc_, chord_, song_reference)

    (mel, acc_, chord_, song_reference) = find_by_length(melody, acc, chord, 6)
    acc_pool[6] = (mel, acc_, chord_, song_reference)

    texture_filter = {}
    for key in acc_pool:
        acc_track = acc_pool[key][1]
        #CALCULATE HORIZONTAL DENSITY
        onset_positions = (np.sum(acc_track, axis=-1) > 0) * 1.
        HD = np.sum(onset_positions, axis=-1) / acc_track.shape[1]
        simu_notes = np.sum((acc_track > 0) * 1., axis=-1)
        VD = np.sum(simu_notes, axis=-1) / (np.sum(onset_positions, axis=-1) + 1e-10)
        dst = np.sort(HD)
        HD_anchors = [dst[len(dst)//5], dst[len(dst)//5 * 2], dst[len(dst)//5 * 3], dst[len(dst)//5 * 4]]
        HD_Bins = [
        HD < HD_anchors[0], 
        (HD >= HD_anchors[0]) * (HD < HD_anchors[1]), 
        (HD >= HD_anchors[1]) * (HD < HD_anchors[2]), 
        (HD >= HD_anchors[2]) * (HD < HD_anchors[3]), 
        HD >= HD_anchors[3]
        ]

        dst = np.sort(VD)
        VD_anchors = [dst[len(dst)//5], dst[len(dst)//5 * 2], dst[len(dst)//5 * 3], dst[len(dst)//5 * 4]]
        VD_Bins = [
        VD < VD_anchors[0], 
        (VD >= VD_anchors[0]) * (VD < VD_anchors[1]), 
        (VD >= VD_anchors[1]) * (VD < VD_anchors[2]), 
        (VD >= VD_anchors[2]) * (VD < VD_anchors[3]), 
        VD >= VD_anchors[3]
        ]
        texture_filter[key] = (HD_Bins, VD_Bins)
    return acc_pool, texture_filter

def select_phrase(query_phrases, melody_queries, query_seg, acc_pool, texture_filter, prefilter, spotlight):
    print('Phrase Selection Begins:\n\t', len(query_phrases), 'phrases in query lead sheet;\n\t', 'Refer to', spotlight, 'as much as possible;\n\t', 'Set note density filter:', prefilter, '.')
    #为节省时间，直接读入边权，而不走模型inference。目前仅支持4小节、6小节，8小节乐句的相互过渡衔接。
    edge_weights=np.load('./data files/edge_weights_0714.npz', allow_pickle=True)
    phrase_indice, chord_shift = new_new_search(
                                            melody_queries, 
                                            query_seg, 
                                            acc_pool, 
                                            edge_weights, 
                                            texture_filter, 
                                            filter_id=prefilter, 
                                            spotlights=ref_spotlight(spotlight))

    path = phrase_indice[0]
    # print("path is", path)
    shift = chord_shift[0]
    # print("shift is", shift)
    reference_set = []
    df = pd.read_excel("./data files/POP909 4bin quntization/four_beat_song_index.xlsx")
    # print("acc pool is", acc_pool)
    # print("query phrases is", query_phrases)
    for idx_phrase, phrase in enumerate(query_phrases):
        phrase_len = phrase[1]
        song_ref = acc_pool[phrase_len][-1]
        # print("len of song ref of ", idx_phrase, phrase, len(song_ref))
        idx_song_ref_loc = path[idx_phrase][0]
        # print("the idx for song ref is", idx_song_ref_loc)
        if len(song_ref) <= idx_song_ref_loc:
            pass
        else:
            idx_song = song_ref[idx_song_ref_loc][0]
            song_name = df.iloc[idx_song][1]
            reference_set.append((idx_song, song_name))
    print('Reference chosen:', reference_set)
    print('Pitch Transpositon (Fit by Model):', shift)
    #uncomment if you want the acc register to be lower or higher
    #for i in range(len(shift)):
    #    if shift[i] > 0: 
    #        shift[i] = shift[i] - 6
    #print('Adjusted Pitch Transposition:', shift)

    # return path[:-1], shift[:-1]
    return path, shift

def save_result_midi(piano_roll, chord_table, acc_pool, query_seg, path, shift):
    time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = time + '.mid'
    save_path = '../output/acco_output/' + name
    print('Generating Output Midi...')
    res_midi = render_acc(piano_roll, chord_table, query_seg, path, shift, acc_pool)
    res_midi.write(save_path)
    print('Output Midi Saved at', save_path)
    return name
