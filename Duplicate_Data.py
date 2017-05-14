from PIL import Image, ImageOps
import os
from os import listdir
from os.path import isfile, join


def mirror_images(File_Path):
    photos = [f for f in listdir(File_Path) if isfile(join(File_Path, f))]
    for infile in photos:
        outfile = os.path.splitext(infile)[0]
        if infile != outfile:
            try:
                img = Image.open(File_Path + infile)
                img = ImageOps.mirror(img)
                img.save(File_Path + "DUP" + outfile + ".png", "PNG")  # save as new image in the original folder

            except IOError as e:
                print("cannot create mirrored image for" +e, infile)


#mirror_images("Data/Raw_Bex/")
#mirror_images("Data/Raw_Peet/")
#mirror_images("Data/Raw_Dad/")
mirror_images("Data/Raw_Joan/")