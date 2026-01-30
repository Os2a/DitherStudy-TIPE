# Etude des Ditherings / Tramages pour TIPE

# Importation de libs
import math, random, sys, os
import pandas as pd
from PIL import Image, ImageDraw  # Manipulation d'images


# Le code sera divisé en plusieur chapitres en fonction du sujet étudié.
sujet = []
def chapter(n):
    return (n == 0 or n in sujet or sujet == [])

# Classe d'image, fonction elementaires
class MyImage:
    def __init__(self, *args): # arg = soit un nom de ficher, soit un couple de dimension 
        if isinstance(args[0], str):
            print(f"[LOAD] {args[0]}")
            self.img = Image.open(args[0])
        else:
            self.img = Image.new("RGB", args[0], color="white")
        self.draw = ImageDraw.Draw(self.img)
    def save(self, name):
        print(f"[PNG] {name}")
        self.img.save(name)

    def size(self):
        return self.img.size

    def getGray(self, x, y):
        r, g, b = self.img.getpixel((x, y))
        return g / 255.0

    def getRgb(self, x, y):
        r, g, b = self.img.getpixel((x, y))
        return [r / 255.0, g / 255.0, b / 255.0]
        

    def setGray(self, x, y, t):
        p = int(t * 255.999)
        self.img.putpixel((x, y), (p, p, p))

    def setRgb(self, x, y, r, g, b):
        r = int(r * 255.999)
        g = int(g * 255.999)
        b = int(b * 255.999)
        self.img.putpixel((x, y), (r, g, b))

    def getRegion(self, x, y, w, h):
        dest = MyImage((w, h))
        dest.img.paste(self.img.crop((x, y, x+w, y+h)))
        return dest

    def getZoom(self, z):
        w, h = self.img.size
        dest = MyImage((w * z, h * z))
        for y in range(h):
            for x in range(w):
                rgb = self.getRgb(x, y)
                for j in range(z):
                    for i in range(z):
                        dest.setRgb(x * z + i, y * z + j, *rgb)
        return dest

    def copyTo(self, dest, pos):
        dest.img.paste(self.img, pos)

# Fonction liée à la correction gamma 
class Gamma:
    def CtoI(x):
        if x < 0:
            return - math.pow(-x, 2.2)
        return math.pow(x, 2.2)
    def ItoC(x):
        if x < 0:
            return - math.pow(-x, 1 / 2.2)
        return math.pow(x, 1 / 2.2)
    CtoI = staticmethod(CtoI)
    ItoC = staticmethod(ItoC)
    def CtoI3(x):
        return [Gamma.CtoI(x[0]), Gamma.CtoI(x[1]), Gamma.CtoI(x[2])]
    def ItoC3(x):
        return [Gamma.ItoC(x[0]), Gamma.ItoC(x[1]), Gamma.ItoC(x[2])]
    CtoI3 = staticmethod(CtoI3)
    ItoC3 = staticmethod(ItoC3)
    def Cto2(x):
        if x < Gamma.CtoI(0.50):
            return 0.
        return 1.
    def Cto3(x):
        if x < Gamma.CtoI(0.25):
            return 0.
        elif x < Gamma.CtoI(0.75):
            return Gamma.CtoI(0.5)
        return 1.
    def Cto4(x):
        if x < Gamma.CtoI(0.17):
            return 0.
        elif x < Gamma.CtoI(0.50):
            return Gamma.CtoI(0.3333)
        elif x < Gamma.CtoI(0.83):
            return Gamma.CtoI(0.6666)
        return 1.
    Cto2 = staticmethod(Cto2)
    Cto3 = staticmethod(Cto3)
    Cto4 = staticmethod(Cto4)

# Crée une matrice
def Matrix(w, h, val=0):
    w = int(w)
    h = int(h)
    return [[val for _ in range(w)] for _ in range(h)]

def MatrixToImage(mat):
    w,h = len(mat),len(mat[0])
    mimg = MyImage((w,h))

    for x,y in rangexy(w,h):
        mimg.setGray(x,y,mat[x][y]/255)
    return mimg


# Solution très pratique pour travailler sur l'enssemble des pixels
def rangexy(w, h):
    for y in range(int(h)):
        for x in range(int(w)):
            yield (x, y)

print("Lancement")  

# Image d'étude (dim 512x512)
# Choix d'image typique pour teste : http://www.cs.cmu.edu/afs/cs/project/cil/ftp/html/v-images.html
lena512 = MyImage("Joffre_400x400.jpg")
(w, h) = lena512.size()


# Convertion de l'image en Gris
# Read the compression FAQ [55] for the rationale behind using the green (raison historique)
# http://www.faqs.org/faqs/compression-faq/part1/section-30.html
# Raison plus rigoureuse https://www.youtube.com/watch?v=fv-wlo8yVhk à 12:12

if chapter(0):
    (w, h) = lena512.size()
    lena512bw = MyImage((w, h))
    for x, y in rangexy(w, h):
        rgb = lena512.getRgb(x, y)
        c = rgb[1]
        lena512bw.setGray(x, y, c)
    lena512bw.save("lena512bw.png")

# Changement de taille d'image
# Explication plus en détail (super interessant) dans le post d'Eric Brasseur http://www.4p8.com/eric.brasseur/gamma.html
def gammascale(src, scale): 
    (w, h) = src.size()  
    dest_w = w // scale  
    dest_h = h // scale
    dest = MyImage((dest_w, dest_h)) 
    for x, y in rangexy(dest_w, dest_h):
        r = g = b = 0.
        for i, j in rangexy(scale, scale):   # Accumulateur de valeur par zone
            if x * scale + i < w and y * scale + j < h: 
                rgb = src.getRgb(x * scale + i, y * scale + j)
                r += Gamma.CtoI(rgb[0])
                g += Gamma.CtoI(rgb[1])
                b += Gamma.CtoI(rgb[2])
        r = Gamma.ItoC(r / (scale * scale))  # Moyenne post correction 
        g = Gamma.ItoC(g / (scale * scale))
        b = Gamma.ItoC(b / (scale * scale))
        dest.setRgb(x, y, r, g, b)  
    return dest

if chapter(0):
    lena256bw = gammascale(lena512bw, 4)
    lena256bw.save("lena256bw.png")
    lena256 = gammascale(lena512, 8)
    lena256.save("lena256.png")

# Creation d'un gradient de gris (greyscale gradient) 32x256, utilie pour mieux visualiser nos effets
if chapter(0):
    grad256bw = MyImage((32, 256))
    for x, y in rangexy(32, 256):
        grad256bw.setGray(x, 255 - y, y / 255.)
    grad256bw.save("gradient256bw.png")


# Gradient de couleur 
if chapter(0):
    grad256 = MyImage((64, 256))
    for x, y in rangexy(64, 256):
        grad256.setRgb(x, y, 255 - x / 63,  (255-y) / 255,  x / 63)
    grad256.save("gradient256.png")


##############################################################################

if chapter(1):
    print("Chapitre 1, Quantisation des couleurs ")

def test11x(src, threshold):
    (w, h) = src.size()
    dest = MyImage((w, h))
    for x, y in rangexy(w, h):
        c = src.getGray(x, y) > threshold # 0/1 : Plus petit ou plus grand que le seuil
        dest.setGray(x, y, c)
    return dest

if chapter(1):
    test11x(grad256bw, 0.5).save("out/1-1-1grad.png")
    test11x(lena256bw, 0.5).save("out/1-1-1lena.png")
    test11x(grad256bw, 0.4).save("out/1-1-2grad.png")
    test11x(lena256bw, 0.4).save("out/1-1-2lena.png")
    test11x(grad256bw, 0.6).save("out/1-1-3grad.png")
    test11x(lena256bw, 0.6).save("out/1-1-3lena.png")

def test12x(src, colors): # Ajout de choix de couleur (teinte de gris)
    (w, h) = src.size()
    dest = MyImage((w, h))
    q = colors - 1 
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        c = math.floor(c * colors) / q # Fonction escalier à l'echelle 
        dest.setGray(x, y, c)
    return dest

if chapter(1):
    test12x(grad256bw, 3).save("out/1-2-1grad.png")
    test12x(lena256bw, 3).save("out/1-2-1lena.png")
    test12x(grad256bw, 5).save("out/1-2-2grad.png")
    test12x(lena256bw, 5).save("out/1-2-2lena.png")

""" Manque d'intéret 
def test12y(src, colors): # Changement sur le choix de la quantisation (arrondit au plus proche) 
    (w, h) = src.size()
    dest = MyImage((w, h))
    q = colors - 1
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        c = math.floor((c + 0.5 / q) * q) / q
        dest.setGray(x, y, c)
    return dest


if chapter(1):
    test12y(grad256bw, 3).save("out/1-2-3grad.png")
    test12y(lena256bw, 3).save("out/1-2-3lena.png")
    test12y(grad256bw, 5).save("out/1-2-4grad.png")
    test12y(lena256bw, 5).save("out/1-2-4lena.png")
"""

def graph_hist(l,t,ep,es):
    dest = MyImage((len(l)*(ep+es), max(l)+es))
    t = [int(x*len(l)) for x in t]
    
    for i,v in enumerate(l) : 
        for dx,dy in rangexy(ep, v) :
            dest.setRgb(i*(ep+es)+dx,dy,255*(i in t),0,0)
    return dest
         
        

# Avec Seuil Dynamique
def test13x(src, n, imhist = ""):
    (w, h) = src.size()
    dest = MyImage((w, h))

    # Création d'un histograme 
    histo = [0] * 256
    for x, y in rangexy(w, h):
        histo[int(src.getGray(x, y) * 255.9999)] += 1

    thresholds = [(1 + i) / n for i in range(n - 1)]
    values = [i / (n - 1) for i in range(n)]
    
    # déoupage histogram
    total = 0
    t = 0
    for i in range(256):
        total += histo[i]
        if total > thresholds[t] * w * h:
            thresholds[t] = i / 255
            t += 1
            if t + 1 > n - 1:
                break
    # graphic de l'histograme
    if (imhist != ""):
        graph_hist(histo,thresholds,5,2).save(f"out/{imhist}_histo.png")
    # Calcule d'image
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        for (i, t) in enumerate(thresholds):
            if c < t:
                dest.setGray(x, y, values[i])
                break
            else:
                dest.setGray(x, y, values[n - 1])
    return dest

if chapter(1):
    test13x(grad256bw, 2,"1-3-1grad").save("out/1-3-1grad.png")
    test13x(lena256bw, 2,"1-3-1lena").save("out/1-3-1lena.png")
    test13x(grad256bw, 5,"1-3-2grad").save("out/1-3-2grad.png")
    test13x(lena256bw, 5,"1-3-2lena").save("out/1-3-2lena.png")


# Output 1.4.2: gaussian random dithering

def test141(src):
    random.seed(0)
    (w, h) = src.size()
    dest = MyImage((w, h))
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        d = c > random.random()
        dest.setGray(x, y, d)
    return dest

def test142(src):
    random.seed(0)
    (w, h) = src.size()
    dest = MyImage((w, h))
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        d = c > random.gauss(0.5, 0.15)
        dest.setGray(x, y, d)
    return dest

if chapter(1):
    test141(grad256bw).save("out/1-4-1grad.png")
    test141(lena256bw).save("out/1-4-1lena.png")

if chapter(1):
    test142(grad256bw).save("out/1-4-2grad.png")
    test142(lena256bw).save("out/1-4-2lena.png")

##############################################################################
if chapter(2):
    print("Chapitre 2, Patern Demi teinte")

if chapter(2): 
    dest = MyImage((320, 80))
    for x in range(320):
        d = 8 >> (x // 80)  # Utilisation amusante des bits pour les paternes
        for y in range(80): 
            c = (x // d + y // d) & 1
            dest.setGray(x, y, c)
    dest.save("out/2-1-1pat.png")

    dest = MyImage((320, 80))
    for x in range(320):
        d = 8 >> (x // 80)
        for y in range(40):
            c = ((x // d + y // d) & 1) or (y // d & 1)
            dest.setGray(x, y, c)
        for y in range(40, 80):
            c = ((x // d + y // d) & 1) and (y // d & 1)
            dest.setGray(x, y, c)
    dest.save("out/2-1-2pat.png")


def test211(src):
    (w, h) = src.size()
    dest = MyImage((w, h))
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        if c < 0.2:
            c = 0.
        elif c < 0.4:
            c = ((x + y) & 1) and (y & 1)
        elif c < 0.6:
            c = (x + y) & 1
        elif c < 0.8:
            c = ((x + y) & 1) or (y & 1)
        else:
            c = 1.
        dest.setGray(x, y, c)
    return dest

if chapter(2):
    test211(grad256bw).save("out/2-1-1grad.png")
    test211(lena256bw).save("out/2-1-1lena.png")


########### 2.3 : Cas des Matrices ###########


# Fonction de trammage pour l'image 
def ordereddither(src, mat):
    (w, h) = src.size()
    dest = MyImage((w, h))
    dx = len(mat[0])
    dy = len(mat)
    for x, y in rangexy(w, h):
        c = src.getGray(x, y)
        threshold = (1. + mat[y % dy][x % dx]) / (dx * dy + 1)
        c = c > threshold
        dest.setGray(x, y, c)
    return dest

# Fonction récursive de création d'une matrice de Bayer 
def makebayer(rank, mat = False):
    if not mat:
        mat = Matrix(1, 1)
    if not rank:
        return mat
    n = len(mat)
    newmat = Matrix(n * 2, n * 2)
    for i, j in rangexy(n, n):
        x = mat[j][i]
        newmat[j * 2][i * 2] = x
        newmat[j * 2][i * 2 + 1] = x + n * n * 3
        newmat[j * 2 + 1][i * 2] = x + n * n * 2
        newmat[j * 2 + 1][i * 2 + 1] = x + n * n
    return makebayer(rank - 1, newmat)


DITHER_BAYER22 = makebayer(1)
DITHER_BAYER44 = makebayer(2)
DITHER_BAYER88 = makebayer(3)

DITHER_CLUSTER44 = \
    [[ 12,  5,  6, 13],
     [  4,  0,  1,  7],
     [ 11,  3,  2,  8],
     [ 15, 10,  9, 14]]

DITHER_CLUSTER88 = \
    [[ 24, 10, 12, 26, 35, 47, 49, 37],
     [  8,  0,  2, 14, 45, 59, 61, 51],
     [ 22,  6,  4, 16, 43, 57, 63, 53],
     [ 30, 20, 18, 28, 33, 41, 55, 39],
     [ 34, 46, 48, 36, 25, 11, 13, 27],
     [ 44, 58, 60, 50,  9,  1,  3, 15],
     [ 42, 56, 62, 52, 23,  7,  5, 17],
     [ 32, 40, 54, 38, 31, 21, 19, 29]]

DITHER_LINE53 = \
    [[  9,  3,  0,  6, 12],
     [ 10,  4,  1,  7, 13],
     [ 11,  5,  2,  8, 14]]



if chapter(2):
    ordereddither(grad256bw, DITHER_BAYER22).save("out/2-3-0grad.png")
    ordereddither(lena256bw, DITHER_BAYER22).save("out/2-3-0lena.png")

    ordereddither(grad256bw, DITHER_BAYER44).save("out/2-3-1grad.png")
    ordereddither(lena256bw, DITHER_BAYER44).save("out/2-3-1lena.png")

    ordereddither(grad256bw, DITHER_BAYER88).save("out/2-3-1bgrad.png")
    ordereddither(lena256bw, DITHER_BAYER88).save("out/2-3-1blena.png")

    ordereddither(grad256bw, DITHER_CLUSTER44).save("out/2-3-2grad.png")
    ordereddither(lena256bw, DITHER_CLUSTER44).save("out/2-3-2lena.png")

    ordereddither(grad256bw, DITHER_CLUSTER88).save("out/2-3-2bgrad.png")
    ordereddither(lena256bw, DITHER_CLUSTER88).save("out/2-3-2blena.png")

    ordereddither(grad256bw, DITHER_LINE53).save("out/2-3-3grad.png")
    ordereddither(lena256bw, DITHER_LINE53).save("out/2-3-3lena.png")


    def makegauss(n): # Récuperation des coefficiant Gaussien 
        c = (-1. + n) / 2
        mat = Matrix(n, n)
        for x, y in rangexy(n, n):
            mat[y][x] = math.exp(- ((c - x) * (c - x) + (c - y) * (c - y)) / (0.05 * n * n)) 
        return mat

    def cblur(x, y, src):
        (w, h) = src.size()
        w -= 1
        h -= 1
        
        weights = makegauss(5)
        
        gb = 0
        for dx,dy in rangexy(5,5):
                vrx = x + dx - 2
                vry = y + dy - 2
                
                # Gestion des bords (mirroring)
                if vrx < 0: vrx = -vrx - 1
                if vry < 0: vry = -vry - 1
                if vrx > w: vrx = 2 * w - vrx + 1
                if vry > h: vry = 2 * h - vry + 1
                
                # Ajouter le poids
                gb += src.getGray(vrx, vry) * (weights[dx][dy]/4)
        
        return gb



    dest = MyImage((130, 65))

    random.seed(0)

    for x,y in rangexy(65,65): # WhiteNoise
        c = random.random()
        dest.setGray(x, y, c)

    for x,y in rangexy(65,65): # Blurred WhiteNoise
        c = cblur(x,y,dest)
        dest.setGray(x+65, y, c)
    dest.save("out/2-4-0pat.png")


    ############ BlueNoise #############


    countones = lambda mat: sum(x for l in mat for x in l if x) # Comptes les 1 

    GAUSS77 = makegauss(7)
    GAUSS99 = makegauss(9)

    
    def getminmax(mat, c): # Recherche des cluster max et min
        min = 9999.
        max = 0.
        h = len(mat)
        w = len(mat[0])
        for x, y in rangexy(w, h):
            if mat[y][x] != c:
                continue
            total = 0.
            for i, j in rangexy(7, 7):
                total += mat[(y + j - 3 + h) % h][(x + i - 3 + w) % w] * GAUSS77[j][i] # Gestion des bords (Torique)
            if total > max:
                (max, max_x, max_y) = (total, x, y)
            if total < min:
                (min, min_x, min_y) = (total, x, y)
        return (min_x, min_y, max_x, max_y)

    def makeuniform(n):
        random.seed(0)
        mat = Matrix(n, n)
        for t in range(n * n // 10):
            x = (int)(random.random() * n)
            y = (int)(random.random() * n)
            mat[y][x] = 1
        while True:
            (_, _, x, y) = getminmax(mat, 1.)
            mat[y][x] = 0.
            (x2, y2, _, _) = getminmax(mat, 0.)
            mat[y2][x2] = 1.
            if x2 == x and y2 == y:
                break
        return mat

    def makevoidandcluster(n):
        vnc = Matrix(n, n)
        # Réduction par rank
        mat = makeuniform(n)
        rank = countones(mat)
        while rank > 0:
            rank -= 1
            (_, _, x, y) = getminmax(mat, 1.)
            mat[y][x] = 0.
            vnc[y][x] = rank
        # Accroissement par rank
        mat = makeuniform(n)
        rank = countones(mat)
        while rank < n * n:
            (x, y, _, _) = getminmax(mat, 0.)
            mat[y][x] = 1.
            vnc[y][x] = rank
            rank += 1

        MatrixToImage(vnc).save(f"out/2-7-1Vnc{n}.png") 
        return vnc

    if chapter(2):
        tmp = makevoidandcluster(14)
        ordereddither(grad256bw, tmp).save("out/2-7-1grad.png")
        ordereddither(lena256bw, tmp).save("out/2-7-1lena.png")
        tmp = makevoidandcluster(25)
        ordereddither(grad256bw, tmp).save("out/2-7-2grad.png")
        ordereddither(lena256bw, tmp).save("out/2-7-2lena.png")


    """ Version nul 
    def blured(src):
            (w,h) = src.size()
            dest = MyImage((w, h))
            for x,y in rangexy(w,h):
                c = cblur(x,y,src)
                dest.setGray(x, y, c)
            return dest
    
    def maxmin(src):
        (w, h) = src.size()
        maxx = 0
        minn = 1
        pose = (0, 0)
        pose2 = (0, 0)
        
        for x, y in rangexy(w, h):
            val = src.getGray(x, y)
            if val > maxx: 
                maxx = val
                pose = (x, y)
            
            if val < minn: 
                minn = val
                pose2 = (x, y)
        
        return pose, pose2
    
    def swap(src, p1, p2):
        (x1, y1) = p1
        (x2, y2) = p2
        c1 = src.getGray(x1, y1)
        c2 = src.getGray(x2, y2)
        src.setGray(x1, y1, c2)
        src.setGray(x2, y2, c1)
        


    def Bluenoise(src=False, max_iter=5000):
        if not src:
            random.seed(0)
            dest = MyImage((64, 64))
            for x, y in rangexy(64, 64):
                dest.setGray(x, y, random.random())
            src = test11x(dest, 0.8)
            src.save("out/2-4-INITBBNpat.png")
        
        prec = None
        for i in range(max_iter + 1):
           
            b = blured(src)
            p1, p2 = maxmin(b)
            swap(src, p1, p2)
            if prec != (p2,p1) and prec != (p,p1) :
                prec = (p1,p2)
            else:
                b.save("out/2-4-0BBNpat.png")
                src.save("out/2-4-0BNpat.png")
                break

            if i % 10 == 0: 
                print(i) 
    Bluenoise()
      """  
    
########### Méthode par diffusion d'erreur ###########
if chapter(3):
    print ("Chapter 3. Error diffusion")

# Différentes méthode :
    # Output 3.0.1: naive error diffusion
    # Output 3.1.1: standard Floyd-Steinberg
    # Output 3.1.2: serpentine Floyd-Steinberg
    # Output 3.2.1: Fan (modified Floyd-Steinberg)
    # Output 3.2.1b: Shiau-Fan 1
    # Output 3.2.1c: Shiau-Fan 2
    # Output 3.2.2: Jarvis, Judice and Ninke
    # Output 3.2.3: Stucki
    # Output 3.2.4: Burkes
    # Output 3.2.5: Sierra
    # Output 3.2.6: Two-line Sierra
    # Output 3.2.7: Sierra's Filter Lite
    # Output 3.2.8: Atkinson

    def test3xx(src, mat, serpentine):
        (w, h) = src.size()
        dest = MyImage((w, h))
        lines = len(mat)
        rows = len(mat[0])
        offset = mat[0].index(-1)
        ey = Matrix(w + rows - 1, lines, 0.)
        for y in range(h):
            ex = [0.] * (rows - offset)
            if serpentine and y & 1:
                xrange = range(w - 1, -1, -1)
            else:
                xrange = range(w)
            for x in xrange:
                # definition du pixel
                c = src.getGray(x, y) + ex[0] + ey[0][x + offset]
                d = c > 0.5
                dest.setGray(x, y, d)
                error = c - d
                # propagation de la première ligne
                for dx in range(rows - offset - 2):
                    ex[dx] = ex[dx + 1] + error * mat[0][offset + 1 + dx]
                ex[rows - offset - 2] = error * mat[0][rows - 1]
                # propagation de la ligne suivante
                if serpentine and y & 1:
                    for dy in range(1, lines):
                        for dx in range(rows):
                            ey[dy][x + dx] += error * mat[dy][rows - 1 - dx]
                else:
                    for dy in range(1, lines):
                        for dx in range(rows):
                            ey[dy][x + dx] += error * mat[dy][dx]
            for dy in range(lines - 1):
                ey[dy] = ey[dy + 1]
            ey[lines - 1] = [0.] * (w + rows - 1)
        return dest

ERROR_NAIVE = \
    [[ -1, 1]]
ERROR_FSTEIN = \
    [[    0.,    -1, 7./16],
     [ 3./16, 5./16, 1./16]]
ERROR_JAJUNI = \
    [[    0.,    0.,    -1, 7./48, 5./48],
     [ 3./48, 5./48, 7./48, 5./48, 3./48],
     [ 1./48, 3./48, 5./48, 3./48, 1./48]]
ERROR_FAN = \
    [[    0.,    0.,    -1, 7./16],
     [ 1./16, 3./16, 5./16,    0.]]
ERROR_SHIAUFAN = \
    [[    0.,    0.,    -1, 8./16],
     [ 2./16, 2./16, 4./16,    0.]]
ERROR_SHIAUFAN2 = \
    [[    0.,    0.,    0.,    -1, 8./16],
     [ 1./16, 1./16, 2./16, 4./16,    0.]]
ERROR_STUCKI = \
    [[    0.,    0.,    -1, 8./42, 4./42],
     [ 2./42, 4./42, 8./42, 4./42, 2./42],
     [ 1./42, 2./42, 4./42, 2./42, 1./42]]
ERROR_BURKES = \
    [[    0.,    0.,    -1, 8./32, 4./32],
     [ 2./32, 4./32, 8./32, 4./32, 2./32]]
ERROR_SIERRA = \
    [[    0.,    0.,    -1, 5./32, 3./32],
     [ 2./32, 4./32, 5./32, 4./32, 2./32],
     [    0., 2./32, 3./32, 2./32,    0.]]
ERROR_SIERRA2 = \
    [[    0.,    0.,    -1, 4./16, 3./16],
     [ 1./16, 2./16, 3./16, 2./16, 1./16]]
ERROR_FILTERLITE = \
    [[   0.,   -1, 2./4],
     [ 1./4, 1./4,   0.]]
ERROR_ATKINSON = \
    [[   0.,   -1, 1./8, 1./8],
     [ 1./8, 1./8, 1./8,   0.],
     [   0., 1./8,   0.,   0.]]


if chapter(3):
    test3xx(grad256bw, ERROR_NAIVE, False).save("out/3-0-1grad.png")
    test3xx(lena256bw, ERROR_NAIVE, False).save("out/3-0-1lena.png")

    test3xx(grad256bw, ERROR_FSTEIN, False).save("out/3-1-1grad.png")
    test3xx(lena256bw, ERROR_FSTEIN, False).save("out/3-1-1lena.png")
    
    test3xx(grad256bw, ERROR_FSTEIN, True).save("out/3-1-2grad.png")
    test3xx(lena256bw, ERROR_FSTEIN, True).save("out/3-1-2lena.png")

    test3xx(grad256bw, ERROR_JAJUNI, False).save("out/3-1-3grad.png")
    test3xx(lena256bw, ERROR_JAJUNI, False).save("out/3-1-3lena.png")

    test3xx(grad256bw, ERROR_FAN, False).save("out/3-2-1grad.png")
    test3xx(lena256bw, ERROR_FAN, False).save("out/3-2-1lena.png")

    test3xx(grad256bw, ERROR_SHIAUFAN, False).save("out/3-2-1bgrad.png")
    test3xx(lena256bw, ERROR_SHIAUFAN, False).save("out/3-2-1blena.png")

    test3xx(grad256bw, ERROR_SHIAUFAN2, False).save("out/3-2-1cgrad.png")
    test3xx(lena256bw, ERROR_SHIAUFAN2, False).save("out/3-2-1clena.png")

    test3xx(grad256bw, ERROR_STUCKI, False).save("out/3-2-3grad.png")
    test3xx(lena256bw, ERROR_STUCKI, False).save("out/3-2-3lena.png")

    test3xx(grad256bw, ERROR_BURKES, False).save("out/3-2-4grad.png")
    test3xx(lena256bw, ERROR_BURKES, False).save("out/3-2-4lena.png")

    test3xx(grad256bw, ERROR_SIERRA, False).save("out/3-2-5grad.png")
    test3xx(lena256bw, ERROR_SIERRA, False).save("out/3-2-5lena.png")

    test3xx(grad256bw, ERROR_SIERRA2, False).save("out/3-2-6grad.png")
    test3xx(lena256bw, ERROR_SIERRA2, False).save("out/3-2-6lena.png")

    test3xx(grad256bw, ERROR_FILTERLITE, False).save("out/3-2-7grad.png")
    test3xx(lena256bw, ERROR_FILTERLITE, False).save("out/3-2-7lena.png")

    test3xx(grad256bw, ERROR_ATKINSON, False).save("out/3-2-8grad.png")
    test3xx(lena256bw, ERROR_ATKINSON, False).save("out/3-2-8lena.png")

if chapter(3):
    tmp = MyImage((128, 128))
    for x, y in rangexy(128, 128):
        tmp.setGray(x, y, 0.90)
    test3xx(tmp, ERROR_FSTEIN, True).getZoom(2).save("out/3-3-2lena.png")
    test3xx(tmp, ERROR_FSTEIN, False).getZoom(2).save("out/3-3-1lena.png")

    
if chapter(4):
    print ("Chapitre 4. Model-based dithering")

def gaussian(n, sigma):
    m = Matrix(n, n, 0.)
    t = 0.
    for x, y in rangexy(n, n):
        i = x - (float)(n - 1.) / 2.
        j = y - (float)(n - 1.) / 2.
        v = math.pow(math.e, - (i * i + j * j) / (2. * sigma * sigma))
        m[y][x] = v
        t += v
    for x, y in rangexy(n, n):
        m[y][x] /= t
    return m

def convolution(src, m):
    (w, h) = src.size()
    dest = MyImage((w, h))
    dy = len(m)
    dx = len(m[0])
    srcmat = [[src.getGray(x, y) for x in range(w)] for y in range(h)]
    for x, y in rangexy(w, h):
        c = t = 0.
        for i, j in rangexy(dx, dy):
            u = i - (dx - 1) / 2
            v = j - (dy - 1) / 2
            if x + u >= w or y + v >= h or x + u < 0 or y + v < 0:
                continue
            c += srcmat[int(y + v)][int(x + u)] * m[j][i]
            t += m[j][i]
        dest.setGray(x, y, c / t)
    return dest

if chapter(4):
    tmp = MyImage("out/3-2-1bgrad.png")
    convolution(tmp, gaussian(11, 1.)).save("out/4-1-1grad.png")
    convolution(tmp, gaussian(11, 2.)).save("out/4-1-2grad.png")
    tmp = MyImage("out/3-2-1blena.png")
    convolution(tmp, gaussian(11, 1.)).save("out/4-1-1lena.png")
    convolution(tmp, gaussian(11, 2.)).save("out/4-1-2lena.png")



def test42x(src):
    random.seed(0)
    (w, h) = src.size()
    dest = MyImage((w, h))
    for x, y in rangexy(w, h):
        c = src.getGray(x, y) + random.random() - 0.5
        d = c > 0.5
        dest.setGray(x, y, d)
    return dest

def test42y(src, dest, hvs):
    threshold = 0.4
    kernel = Matrix(8, 8, 0.) 
    for i, j in rangexy(6, 6):
         kernel[j][i] = hvs(i * i + j * j)
    (w, h) = src.size()
    srcmat = Matrix(w, h, 0.)
    destmat = Matrix(w, h, 0.)
    for x, y in rangexy(w, h):
        srcmat[y][x] = src.getGray(x, y)
        destmat[y][x] = dest.getGray(x, y)
    srchvs = Matrix(w, h, 0.)
    desthvs = Matrix(w, h, 0.)
    for x, y in rangexy(w, h):
        srcp = destp = 0.
        for j in range(-5, 6):
            if y + j < 0 or y + j >= h:
                continue
            for i in range(-3, 4):
                if x + i < 0 or x + i >= w:
                    continue
                m = kernel[abs(j)][abs(i)]
                srcp += m * srcmat[y + j][x + i]
                destp += m * destmat[y + j][x + i]
        srchvs[y][x] = srcp
        desthvs[y][x] = destp
    swaps = toggles = 0
    for x, y in rangexy(w, h):
        d = destmat[y][x]
        best = 0.
        # Calcule des effets de l'inverse
        e = 0.
        for j in range(-5, 6):
            if y + j < 0 or y + j >= h:
                continue
            for i in range(-5, 6):
                if x + i < 0 or x + i >= w:
                    continue
                m = kernel[abs(j)][abs(i)]
                p = srchvs[y + j][x + i]
                q1 = desthvs[y + j][x + i]
                q2 = q1 - m * d + m * (1. - d)
                e += abs(q1 - p) - abs(q2 - p)
        if e > best:
            best = e
            op = False
        # Calcule des effets de l'échange
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            if y + dy < 0 or y + dy >= h or x + dx < 0 or x + dx >= w:
                continue
            d2 = destmat[y + dy][x + dx]
            if d2 == d:
                continue
            e = 0.
            for j in range(-6, 7):
                for i in range(-6, 7):
                    if y + j < 0 or y + j >= h or x + i < 0 or x + i >= w:
                        continue
                    ma = kernel[abs(j)][abs(i)]
                    mb = kernel[abs(j - dy)][abs(i - dx)]
                    p = srchvs[y + j][x + i]
                    q1 = desthvs[y + j][x + i]
                    q2 = q1 - ma * d + ma * d2 - mb * d2 + mb * d
                    e += abs(q1 - p) - abs(q2 - p)
            if e > best:
                best = e
                op = (dx, dy)
        # Apply the change if interesting
        if best <= 0.:
            continue
        if op:
            dx, dy = op
            d2 = destmat[y + dy][x + dx]
            destmat[y + dy][x + dx] = d
        else:
            d2 = 1. - d
        destmat[y][x] = d2
        for j in range(-5, 6):
            for i in range(-5, 6):
                m = kernel[abs(j)][abs(i)]
                if y + j >= 0 and y + j < h and x + i >= 0 and x + i < w:
                    desthvs[y + j][x + i] -= m * d
                    desthvs[y + j][x + i] += m * d2
                if op and y + dy + j >= 0 and y + dy + j < h \
                   and x + dx + i >= 0 and x + dx + i < w:
                    desthvs[y + dy + j][x + dx + i] -= m * d2
                    desthvs[y + dy + j][x + dx + i] += m * d
    for x, y in rangexy(w, h):
        dest.setGray(x, y, destmat[y][x])
    return dest

if chapter(4):
    hvs = lambda x: math.pow(math.e, - math.sqrt(x))

    tmp = test42x(grad256bw)
    tmp.save("out/4-2-1grad.png")
    tmp = test42y(grad256bw, tmp, hvs)
    tmp.save("out/4-2-2grad.png")
    tmp = test42y(grad256bw, tmp, hvs)
    tmp.save("out/4-2-3grad.png")
    for n in range(3): tmp = test42y(grad256bw, tmp, hvs)
    tmp.save("out/4-2-4grad.png")

    tmp = test42x(lena256bw)
    tmp.save("out/4-2-1lena.png")
    tmp = test42y(lena256bw, tmp, hvs)
    tmp.save("out/4-2-2lena.png")
    tmp = test42y(lena256bw, tmp, hvs)
    tmp.save("out/4-2-3lena.png")
    for n in range(3): tmp = test42y(lena256bw, tmp, hvs)
    tmp.save("out/4-2-4lena.png")


if chapter(4):
    hvs = lambda x: math.pow(math.e, -x / 8.) + 2. * math.pow(math.e, -x / 1.5)
    tmp = test42x(grad256bw)
    for n in range(5): tmp = test42y(grad256bw, tmp, hvs)
    tmp.save("out/4-2-8grad.png")
    tmp = test42x(lena256bw)
    for n in range(5): tmp = test42y(lena256bw, tmp, hvs)
    tmp.save("out/4-2-8lena.png")


if chapter(5):
    print("Chapitre 5, Benchmark")

if chapter(5):
        
    def rmse(gray, dither):
        (w, h) = gray.size()
        error = 0.
        for y in range(5, h - 5):
            for x in range(5, w - 5):
                c = gray.getGray(x, y)
                d = dither.getGray(x, y)
                error += (c - d) * (c - d)
        return math.sqrt(error / ((w - 10) * (h - 10)))

    hvs_f = lambda x: math.pow(math.e, -x / 8.) + 2. * math.pow(math.e, -x / 1.5)

    def hvs_kernel(size, hvs_fn):
        kernel = Matrix(size, size, 0.)
        center = size // 2
        for x,y in rangexy(size,size):
            dx = x - center
            dy = y - center
            d2 = dx * dx + dy * dy
            kernel[y][x] = hvs_fn(d2)
        total = sum(sum(row) for row in kernel)
        for x,y in rangexy(size,size):
                kernel[y][x] /= total
        return kernel

    lena_images = [
        ("1-1-1","Seuil 1/2"), ("1-1-2","Seuil 1/3"), ("1-1-3","Seuil 2/3"), ("1-2-1","3 Color"), ("1-2-2","5 Couleur"), ("1-3-1","Seuil Dynamique"), ("1-3-2","Seuil Dynamique 5 couleur"), ("1-4-1","Random dithering"), ("1-4-2","gaussian distribution"),
        ("2-1-1","Demi-Teinte grille"), ("2-3-1","Bayer Matrix"), ("2-3-2","Demi-Teinte à point"), ("2-3-3","Demi-Teinte Verticale"),
        ("2-7-1","BlueNoise"),("3-0-1","Diffusion d'Erreur Naif"), ("3-1-1","Floyd-Steinberg"), ("3-1-3","Jarvis, Judice and Ninke"), ("3-2-1","Fan dithering"), ("3-2-3","Stucki dithering"), ("3-2-4","Burkes dithering"), ("3-2-5"," Sierra dithering"),("3-2-6","Two-row Sierra"), ("3-2-7","Filter Lite"),
        ("3-2-8","Atkinson"),("4-1-1","Bayer Flou σ = 1"), ("4-1-2","Bayer Flou σ = 2"), ("4-2-1","DBS étape 0"), ("4-2-2","DBS étape 1"),("4-2-3","DBS étape 2"), ("4-2-4","DBS étape 5"), ("4-2-8","DBS étape 8")
    ]

    gauss_range = 11
    data = {"Méthode": [x[1] for x in lena_images],"σ = 1.0": [], "σ = 1.5":[],"σ = 2.0":[]}
    for sigma in [1.,1.5,2.]:
        hvs_f = lambda x2y2: math.exp(-x2y2 / (2 * sigma * sigma))
        hvs_k = hvs_kernel(gauss_range,hvs_f)
        tmp_ref = convolution(lena256bw, hvs_k)
        for filename, _ in lena_images:
            image_path = f"out/{filename}lena.png"
            image = MyImage(image_path)
            error = rmse(convolution(image, hvs_k), tmp_ref )
            data[f"σ = {sigma}"].append(error)
    
    df = pd.DataFrame(data)
    df.to_excel("Benchmark.xlsx", index=False)
    print(f"Fichier Benchmark exporté")