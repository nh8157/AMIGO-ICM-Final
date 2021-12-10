from django import forms
import glob
import os

os.chdir("instrumentation/instrumentation")
from instrumentation.instrumentation import SONG_SAMPLE, SPOTLIGHT
os.chdir("../../../")

class MusicForm(forms.Form):
    sample_path = "./instrumentation/instrumentation/accompany/"
    sample_choices = (
        (name, (name + ' (' + str(SONG_SAMPLE[name][-1]) + ' Bars)')) for name in SONG_SAMPLE.keys()
    )
    acco_choices = (
        (name, name) for name in SPOTLIGHT
    )
    arra_choices = (
        ('bach', 'Bach'),
        ('nes', 'Nintendo Music'),
        ('musicnet', 'String Quartet'),
        ('lmd', 'Pop'),
    )

    midi_or_sample = forms.ChoiceField(
        label="Do you have a midi file ready?",
        choices=(("Yes", "Yes"), ("No", "No")),
        widget=forms.RadioSelect,
        required=True
        )
    self_midi_file = forms.FileField(
        label="If yes, please upload your midi file",
        required=False,
        )
    sample_midi_file = forms.ChoiceField(
        label="Or you can try a sample midi file",
        choices=sample_choices,
        widget=forms.RadioSelect,
        required=False
        )
    self_segmentation = forms.CharField(
        label="(Optional) Please input your segmentation pattern (example: A4B4):",
        required=False,
        widget=forms.TextInput(attrs={'placeholder': 'A8B8A8B8'})        
        )
    acco_style = forms.MultipleChoiceField(
        label="Please select the accompaniment style(s)", 
        choices=acco_choices, 
        widget=forms.CheckboxSelectMultiple,
        required=True
        )
    arra_style = forms.ChoiceField(
        label="Please select an arrangement style",
        choices=arra_choices,
        widget=forms.RadioSelect,
        required=True
        )