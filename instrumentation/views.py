import os
from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from muspy.inputs import midi
from . import forms
import glob

os.chdir("instrumentation/instrumentation")
from instrumentation.instrumentation import generate, SONG_ROOT
os.chdir("../../")

UPLOAD_PATH = 'uploads'

# Create your views here.
def index(request):
    # return the form
    if request.method == 'POST':
        form = forms.MusicForm(request.POST, request.FILES)
        if form.is_valid():
            # parse each data field from the form
            data = form.cleaned_data
            midi_or_sample = data['midi_or_sample']
            acco_style = data['acco_style']
            arra_style = data['arra_style']
            self_segmentation = data['self_segmentation']
            if midi_or_sample == 'Yes':
                # user uploads his own midi
                midi_file = request.FILES['self_midi_file']
                # check if the file ends with .midi
                extension = midi_file.name.split('.')[-1]
                print(extension)
                if extension != 'mid' and extension != 'MID':
                    # invalid file name
                    return HttpResponse("invalid file: please upload a midi file")
                # save the file to disk
                fs = FileSystemStorage('instrumentation/' + UPLOAD_PATH)
                song_name = midi_file.name
                print(glob.glob('instrumentation/' + UPLOAD_PATH))
                if song_name not in glob.glob('instrumentation/' + UPLOAD_PATH):
                    fs.save(midi_file.name, midi_file)
                midi_path = os.getcwd() + '/instrumentation/' + UPLOAD_PATH
                print(song_name)
                # interface here to connect with instrumentation
                name = generate(midi_path=midi_path, song_name=song_name, acco_style=acco_style, arra_style=arra_style, audio_path='../../static/audio/', uploaded_midi=True, segmentation=self_segmentation)
            else:
                midi_path = os.getcwd() + '/instrumentation/instrumentation/' + SONG_ROOT
                song_name = data['sample_midi_file']
                # interface here to connect with instrumentation
                name = generate(midi_path=midi_path, song_name=song_name, acco_style=acco_style, arra_style=arra_style, audio_path='../../static/audio/')

                
            # render the audio file directly through a html page
            return render(request, 'html/index.html', {'form': form, 'audio': True, 'path': 'audio/' + name}) 
    else:
        form = forms.MusicForm()
    
    return render(request, 'html/index.html', {'form': form, 'audio': False}) 