# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 17:37:59 2017

@author: cbothore
"""
# https://fr.wikipedia.org/wiki/Windows_bitmap

# A first method to parse a BMP file
# It is a binary file
# Import a module to convert bytes from binary files 
# to H (unsigned short, 2 bytes), I (unsigned int, 4 bytes)
import struct

input_filename="/Users/Isidore/Documents/TB/2A/ELU501/Challenge3/population-density-map.bmp"

bmp = open(input_filename, 'rb') # open a binary file
print('-- First part of the header, information about the file (14 bytes)')
print('Type:', bmp.read(2).decode())
print('Size: %s' % struct.unpack('I', bmp.read(4)))
print('Reserved 1: %s' % struct.unpack('H', bmp.read(2)))
print('Reserved 2: %s' % struct.unpack('H', bmp.read(2)))
offset=struct.unpack('I', bmp.read(4))
print('Image start after Offset: %s' % offset)

print('-- Second part of the header, DIB header, bitmap information header (varying size)')
print('The size of this DIB Header Size: %s' % struct.unpack('I', bmp.read(4)))
print('Width: %s' % struct.unpack('I', bmp.read(4)))
print('Height: %s' % struct.unpack('I', bmp.read(4)))
print('Colour Planes: %s' % struct.unpack('H', bmp.read(2)))
pixel_size=struct.unpack('H', bmp.read(2))
print('Bits per Pixel: %s' % pixel_size)
print('Compression Method: %s' % struct.unpack('I', bmp.read(4)))
print('Raw Image Size: %s' % struct.unpack('I', bmp.read(4)))
print('Horizontal Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Vertical Resolution: %s' % struct.unpack('I', bmp.read(4)))
print('Number of Colours: %s' % struct.unpack('I', bmp.read(4)))
print('Important Colours: %s' % struct.unpack('I', bmp.read(4)))

# At this step, we have read 14+40 bytes
# As offset[0] = 54, from now, we will read the BMP content
# You have to read each pixel now, and do what you have to do
# First pixel is bottom-left, and last one top-right
# .........
bmp.close()


# Another method to parse a BMP image
# To manipulate imageIf you want to work with image data in Python, 
# numpy is the best way to store and manipulate arrays of pixels. 
# You can use the Python Imaging Library (PIL) to read and write data 
# to standard file formats.

# Use PIL module to read file
# http://pillow.readthedocs.io/en/latest/
from PIL import Image
import numpy as np
im = Image.open(input_filename)

# This modules gives useful informations
width=im.size[0]
heigth=im.size[1]
colors = im.getcolors(width*heigth)
print('Nb of different colors: %d' % len(colors))
# To plot an histogram
from matplotlib import pyplot as plt
def hexencode(rgb):
    r=rgb[0]
    g=rgb[1]
    b=rgb[2]
    return '#%02x%02x%02x' % (r,g,b)

for idx, c in enumerate(colors):
    plt.bar(idx, c[0], color=hexencode(c[1]))

plt.show()
# We have 32 different colors in this image
# We can see that we have "only" 91189 black pixels able to stop zombies 
# but we have a large majority of dark ones slowing their progression

# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(im) 

print(p.shape)
# a result (3510, 4830, 3) means (rows, columns, color channels)
# where 3510 is the height and 4830 the width

# to get the Red value of pixel on row 3 and column 59
p[3,59][0]

# How to get the coordinates of the green and red pixels where 
# (0,0) is top-left and (width-1, height-1) is bottom-right
# In numpy array, notice that the first dimension is the height, 
# and the second dimension is the width. That is because, for a numpy array, 
# the first axis represents rows (our classical coord y), 
# and the second represents columns (our classical x).

# First method
# Here is a double loop (careful, O(n²) complexity) to parse the pixels from
# (0,0) top-left and (heigth-1, width-1) is bottom-right
#for y in range(heigth):
#    for x in range(width):
#        # p[y,x] is the coord (x,y), x the colum, and y the line
#        # As an exemple, we search for the green and red pixels
#        # p[y,x] is an array with 3 values
#        # We test if there is a complete match between the 3 values 
#        # from both arrays p[y,x] and np.array([0,255,0])
#        # to detect green pixels
#       if (p[y,x] == np.array([0,255,0])).all():
#            print("Coordinates (x,y) of the green pixel: (%s,%s)" % (str(x),str(y)))
#            # Coordinates (x,y) of the green pixel: (4426,2108)
#        if (p[y,x] == np.array([255,0,0])).all():
#            print("Coordinates (x,y) of the red pixel: (%s,%s)" % (str(x),str(y)))
#            # Coordinates (x,y) of the red pixel: (669,1306)

# Here is a more efficient method to get the location of the green and red pixels
#mask = np.all(p == (0, 255, 0), axis=-1)
#z = np.transpose(np.where(mask))
#print("Coordinates (x,y) of the green pixel: (%d,%d)" % (z[0][1],z[0][0]))
#mask = np.all(p == (255, 0, 0), axis=-1)
#z = np.transpose(np.where(mask))
#print("Coordinates (x,y) of the red pixel: (%d,%d)" % (z[0][1],z[0][0]))


# Now we have the source and the target positions of our zombies
# we could convert our RGB image into greyscale image to manipulate
# only 1 value for the color and deduce more easily the density of
# population
grayim = im.convert("L")
grayim.show()
colors = grayim.getcolors(width*heigth)
print('Nb of different colors: %d' % len(colors))
# With the image im, let's generate a numpy array to manipulate pixels
p = np.array(grayim) 
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
plt.hist(p.ravel())

# from gray colors to density
density = p/255.0
# plot the histogram. We still have a lot of dark colors. Just to check ;-)
plt.hist(density.ravel())

# We can use the gray 2D array density to create our graph
# Gray colors density[y,x] range now from 0 (black) to 1 (white)
# density[0,0] is top-left pixel density
# and density[heigth-1,width-1] is bottom-right pixel

# Population - Rize : 107 400 

def cells(image) : # x et y correspondent à un gros pixel (15x15) # definit les cellules dans le graph --> densité de population, élévation, 
    width=image.size[0] #4830
    height=image.size[1] #3510
    w=width//15
    h=height//15
    matrice=[]
    population=np.zeros((h,w))
    for y in range(w):
        for x in range(h):
            x1=15*x
            x2=15*x+14
            y1=y*15
            y2=y*15+14
            matrice=density[x1:x2,y1:y2]
            population[x][y]=matrice.sum()
    return population

Human=cells(im)        
 # Coordinates (y,x) of the green pixel: (4426,2108)
width=im.size[0]
height=im.size[1]
w=width//15 
h=height//15
densityZom=np.zeros((height,width))
densityZom[2108][4426]=570

def cells2(densityZom, image) : # x et y correspondent à un gros pixel (15x15) # definit les cellules dans le graph --> densité de population, élévation, 
    width=image.size[0]
    height=image.size[1]
    w=width//15
    h=height//15
    im=[]
    population=np.zeros((h,w))
    for y in range(w):
        for x in range(h):
            x1=15*x
            x2=15*x+14
            y1=y*15
            y2=y*15+14
            im=densityZom[x1:x2,y1:y2]
            population[x][y]=im.sum()
    return population
    
Zombies= cells2(densityZom,im) 
altitude=np.ones((h,w))


    
def step1(zombies,human) : # cells : (zombies, human, altitude)
    #return the number of zombies on j+1
    width=len(human[0])
    height=len(human)
    zombies2=zombies.copy()
    for y in range(5,width-5): #Pas besoin d'étudier les bords de la carte
        for x in range(5,height-5):
            if zombies[x][y]!=0:
                totHumans=human[x-1][y-1]+human[x-1][y]+human[x-1][y+1]+human[x][y-1]+human[x][y+1]+human[x+1][y-1]+human[x+1][y]+human[x+1][y+1]
                if altitude[x-1][y-1]<20 :
                    zombies2[x-1][y-1]=human[x-1][y-1]/totHumans*zombies[x][y]*altitude[x-1][y-1]
                    
                if altitude[x-1][y]<20 :
                    zombies2[x-1][y]=human[x-1][y]/totHumans*zombies[x][y]*altitude[x-1][y]
                    
                if altitude[x-1][y+1]<20 :
                    zombies2[x-1][y+1]=human[x-1][y+1]/totHumans*zombies[x][y]*altitude[x-1][y+1]
                    
                if altitude[x][y-1]<20 :
                    zombies2[x][y-1]=human[x][y-1]/totHumans*zombies[x][y]*altitude[x][y-1]
                    
                if altitude[x][y+1]<20 :
                    zombies2[x][y+1]=human[x][y+1]/totHumans*zombies[x][y]*altitude[x][y+1]
                    
                if altitude[x+1][y-1]<20 :
                    zombies2[x+1][y-1]=human[x+1][y-1]/totHumans*zombies[x][y]*altitude[x+1][y-1]
                    
                if altitude[x+1][y]<20 :
                    zombies2[x+1][y]=human[x+1][y]/totHumans*zombies[x][y]*altitude[x+1][y]
                    
                if altitude[x+1][y+1]<20 :
                    zombies2[x+1][y+1]=human[x+1][y+1]/totHumans*zombies[x][y]*altitude[x+1][y+1]
                zombies2[x][y]=0
    return zombies2
            
def step2(zombies,human) : 
    # retourne le nombre de zombies restants
    width=len(human[0])
    height=len(human)
    for y in range(5,width-5): 
        for x in range(5,height-5):
            hum=human[x,y]
            zom=zombies[x,y]
            canKill=10*zom
            if hum>canKill:
                human[x,y]-=canKill
                zombies[x,y]+=canKill
            else:
                zombies[x,y]+=hum
                human[x,y]=0
    return (zombies,human)

def step3(zombies,human):
     # retourne le nombre d'humains restants
    width=len(human[0])
    height=len(human)
    for y in range(5,width-5):
        for x in range(5,height-5):
           hum=human[x][y]
           zom=zombies[x][y]
           canKill=10*hum
           if zom>canKill:
                human[x,y]+=canKill
                zombies[x,y]-=canKill
           else:
                zombies[x,y]=0
                human[x,y]+=zom
    return (zombies,human)
    
def zomMortality(zombies):
    
    return None

# Coordinates (y,x) of the red pixel: (669,1306)
 
def apocalypse(zombies,human):
    red_x=141//15
    red_y=297//15

    days=0
    while zombies[red_x][red_y]==0:
        zombies=step1(zombies,human)
        print("step 1 is ", "z:", zombies)
        (zombies,human)=step2(zombies,human)
        print("step  2 is ", "z:", zombies,"h:", human )
        (zombies,human)=step3(zombies,human)
        print("step  3 is ", "z:", zombies,"h:", human )
        days+=1
        print("number of days are", days)
    return days


print("days are", apocalypse(Zombies,Human))
#test=step1(Zombies,Human)
#testZ,testH=step2(test,Human)
#testZ[139:142, 294:297]
#testH[139:142, 294:297]
#    
