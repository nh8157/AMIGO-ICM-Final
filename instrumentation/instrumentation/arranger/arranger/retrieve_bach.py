import shutil
import music21.corpus

for path in music21.corpus.getComposer("bach"):
    print(path)
    if path.suffix in (".mxl", ".xml"):
        shutil.copyfile(path, "data/bach/raw/" + path.name)
