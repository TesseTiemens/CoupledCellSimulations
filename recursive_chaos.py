import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
import cv2
import matplotlib as mpl
from progress.bar import Bar
from datetime import datetime

def cellamount(n):
  num = 0
  ncount = n

  #finding the amount of points for this amount of layers
  while ncount >= 0:
    num += 3 * 2**ncount
    ncount -= 1
  return num

def n_matrixgen(n,couplconst = 1):
  #finding the amount of points for this amount of layers
  num = cellamount(n)
  #negating the coupling contant for further use
  couplconst = -couplconst
  #setting up the self-coupling and center triangle
  n_matrix = np.identity(num)
  n_matrix[0,1] = couplconst
  n_matrix[0,2] = couplconst
  n_matrix[1,0] = couplconst
  n_matrix[2,0] = couplconst

  #filling in the rest of the matrix
  for i in range(num):
    for j in range(num):
      if ((abs(i-j)==1 and np.mod(min(i,j),2)==1) #within own triangle
          or i-(2*j+2) ==1 or i-(2*j+2) ==2 #outward coupling
          or j-(2*i+2) ==1 or j-(2*i+2) ==2): #inward coupling
        n_matrix[i,j] = couplconst
  return n_matrix

#generating all the coordinates
def coordfractal(level, maxlevel, number, angle, mother, coordarray,scalefactor = 0.8):
  '''
  level: current leven, maxlevel: maximum level, number: current cell number,
  angle: angle from norma, mother: coordinate previous cell, coordarray: should
  be an empty array of the right length in first call, scalefactor: amont the
  distances get scaled with per level 
  '''
  owncoord = [mother[0]+np.cos(angle)*scalefactor**level,mother[1]+np.sin(angle)*scalefactor**level]
  coordarray[number-1,:] = owncoord

  if level < maxlevel:
    #left
    coordarray = coordfractal(level+1, maxlevel, 2*number+2, angle - np.pi/6, owncoord, coordarray)
    #right
    coordarray = coordfractal(level+1, maxlevel, 2*number+3, angle +
                              np.pi/6, owncoord,
                              coordarray)
  return coordarray

def coordhelper(maxlevel, scalefactor = 0.8):
  num = cellamount(maxlevel)
  array = np.empty((num,2))
  #top
  array = coordfractal(0,maxlevel,1,np.pi/2,[0,0],array,scalefactor=scalefactor)
  #left
  array = coordfractal(0,maxlevel,2,-np.pi/6,[0,0],array,scalefactor=scalefactor)
  #right
  array = coordfractal(0,maxlevel,3,7*np.pi/6,[0,0],array,scalefactor=scalefactor)
  return array

def couplinglines(ax,coordarray,couplmatrix,num): #draws lines between points
  for i in range(num):
    for j in range(num):
      if couplmatrix[i,j] != 0:
        x_vals = [coordarray[i,0],coordarray[j,0]]
        y_vals = [coordarray[i,1],coordarray[j,1]]
        ax.plot(x_vals,y_vals,c = 'k')
  return ax

def videomaker(cellnr, levels, tEnd,dt, phasearray,couplmatrix,current_time, scalefactor = 0.8, step = 10, width = 1080, height = 1080, dpi = 100):
  locs =  coordhelper(levels, scalefactor)#dividing a circle into even intervals
  cmap = mpl.cm.get_cmap('seismic') #color map
  # create OpenCV video writer
  video = cv2.VideoWriter('{}.avi'.format(current_time), cv2.VideoWriter_fourcc(*'MP42'), 30, (width,height))
  bar2 = Bar('making video', max = tEnd*step)
  # loop over your images
  for i in range(tEnd*step):

    fig = plt.figure(figsize=(width/dpi,height/dpi), dpi = dpi)
    ax = fig.add_subplot(1,1,1)
    plt.xlim((-3.5,3.5))
    plt.ylim((-3.5,3.5))
    for loc in range(cellnr): #we make a cirlce from the locs array
      color = cmap((1+np.sin(phasearray[int(i/(dt*step)),loc]))*0.5) #getting the color for this circle
      circle = plt.Circle(locs[loc,:],0.1,color=color) #making a circle
      ax.add_artist(circle)
    ax = couplinglines(ax, locs,couplmatrix, cellnr)
    # put pixel buffer in numpy array
    canvas = FigureCanvas(fig)
    canvas.draw()
    mat = np.frombuffer(canvas.tostring_rgb(), dtype = 'uint8').reshape(height,width,3)
    #mat = np.array(canvas.renderer._renderer)
    #mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)

    # write frame to video
    video.write(mat)
    plt.close()
    bar2.next()

  # close video writer
  bar2.finish()
  cv2.destroyAllWindows()
  video.release()

def recursivecell(levels = 3, init = 0,tEnd = 100,dt = 0.001,couplcoeff = 0.3,drive = 0.5, 
                  vidja = True, scalefactor = 0.8, step = 10, width = 1080, height = 1080, dpi = 100):#video parameters
  #finding amount of cells
  cellnr =cellamount(levels) 
   
  #random startvector
  if init == 0:
    startvect = np.empty((cellnr,1))
    for i in range(cellnr):
      startvect[i,0] = random.random()*2*np.pi
  else:
      startvect = np.array(init)
      startvect = np.reshape(startvect,(cellnr,1))
  initcon = np.array2string(np.reshape(startvect,(1,cellnr)))
  #coupling matrices
  totcoupl = n_matrixgen(levels,couplcoeff)#using function to generate coupling matrix
  drivevect = np.ones(cellnr)*drive
  
  #simulation setup
  phasevect = startvect
  lensarray = np.empty(1)
  lensarray[0] = np.linalg.norm(phasevect)# array of lengths
  t = 0
  print(totcoupl)

  #phasearray = np.array([phasevect],) #total array of all the phases
  phasearray = np.empty((cellnr,int(tEnd/dt + 1)))
  phasearray[:,0] = phasevect[:,0]

  bar1 = Bar('simulating', max = int(tEnd/dt))

  #actually simulating
  while t <tEnd:
    dphase = couplcoeff*(np.mod(-np.matmul(totcoupl,phasevect)+np.pi,np.pi*2)-np.pi)+drivevect #calculating derivative
    phasevect = phasevect+ dt*dphase
    lensarray = np.append(lensarray, np.linalg.norm(phasevect))
    #phasearray = np.append(phasearray, [phasevect], axis=0)
    phasearray[:,int((t/dt)+1)] = phasevect[:,0]
    t = t+dt
    bar1.next()
  bar1.finish()
  
  #transposing the array bc I didn't want to rewrite half the plots
  phasearray = np.transpose(phasearray)
  #finding time for filenames
  now = datetime.now()

  #plot phase stuff
  tarr = np.arange(0,tEnd+1*dt, dt)
  plt.figure(figsize=(15,10))
  for i in range(cellnr):
      plt.plot(tarr, np.mod(phasearray[:,i],2*np.pi))
      plt.savefig('phase_{}.png'.format(now))

  #plot sine
  plt.figure(figsize=(15,5))
  for i in range(cellnr):
      plt.plot(tarr, np.sin(phasearray[:,i]))

  plt.savefig('sine_{}.png'.format(now))

  #Video stuff
  if vidja == True:
    videomaker(cellnr,levels,tEnd,dt, phasearray,totcoupl,now,
               scalefactor,step,width,height,dpi)

