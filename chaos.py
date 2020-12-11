import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
import cv2
import matplotlib as mpl
from progress.bar import Bar


def cellring(init = 0,tend = 100, vidja = True):
  #defining variables
  cellnr = 12
  drive = 0.5
  dt = 0.001
  couplcoeff = 0.3
  tEnd = tend 
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
  intcoupl = np.identity(cellnr)
  extcoupl = np.empty((cellnr,cellnr))
  drivevect = np.ones(cellnr)*drive

  for i in range(cellnr):
    for j in range (cellnr):
      if np.mod(np.absolute(i-j),10) == 1:
        extcoupl[i,j] = -0.5
      else:
        extcoupl[i,j]=0
  #simulation setup
  phasevect = startvect
  lensarray = np.empty(1)
  lensarray[0] = np.linalg.norm(phasevect)# array of lengths
  t = 0
  totcoupl = (intcoupl+extcoupl) #full coupling matrix
  print(totcoupl)

  #phasearray = np.array([phasevect],) #total array of all the phases
  phasearray = np.empty((12,int(tEnd/dt + 1)))
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
  #plot phase stuff
  tarr = np.arange(0,tEnd+1*dt, dt)
  plt.figure(figsize=(15,10))
  for i in range(cellnr):
      plt.plot(tarr, np.mod(phasearray[:,i],2*np.pi))
  plt.savefig('phase_{}.png'.format(initcon))

  #plot sine
  plt.figure(figsize=(15,5))
  for i in range(cellnr):
      plt.plot(tarr, np.sin(phasearray[:,i]))

  plt.savefig('sine_{}.png'.format(initcon))




  #Video stuff
  if vidja == True:
    locs = np.arange(0,2*np.pi,2*np.pi/cellnr) #dividing a circle into even intervals
    cmap = mpl.cm.get_cmap('seismic') #color map
    step = 10
    width = 1080
    height = 1080
    dpi = 100
    # create OpenCV video writer
    video = cv2.VideoWriter('{}.avi'.format(initcon), cv2.VideoWriter_fourcc(*'MP42'), 30, (width,height))
    bar2 = Bar('making video', max = tEnd*step)
    # loop over your images
    for i in range(tEnd*step):

      fig = plt.figure(figsize=(width/dpi,height/dpi), dpi = dpi)
      ax = fig.add_subplot(1,1,1)
      plt.xlim((-1.5,1.5))
      plt.ylim((-1.5,1.5))
      for loc in range(cellnr): #we make a cirlce from the locs array
        color = cmap((1+np.sin(phasearray[int(i/(dt*step)),loc]))*0.5) #getting the color for this circle
        circle = plt.Circle((np.cos(locs[loc]),np.sin(locs[loc])),0.1,color=color) #making a circle
        ax.add_artist(circle)

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

