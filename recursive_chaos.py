import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.cm as cm
import cv2
import matplotlib as mpl
from progress.bar import Bar
from datetime import datetime
import multiprocessing as mp
import time
import queue
import math
import vapeplot as vp

def cellamount(n):
  num = 0
  ncount = n

  #finding the amount of points for this amount of layers
  while ncount >= 0:
    num += 3 * 2**ncount
    ncount -= 1
  return num

def n_matrixgen(n,couplconst = 1,prnt = False):
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
  
  outercells =3* 2**n #determining how many cells are on the outer layer
  #filling in the rest of the matrix
  for i in range(num):
    for j in range(num):
      if ((abs(i-j)==1 and np.mod(min(i,j),2)==1) #within own triangle
          or i-(2*j+2) ==1 or i-(2*j+2) ==2 #outward coupling
          or j-(2*i+2) ==1 or j-(2*i+2) ==2): #inward couplingi
        if i>= num-outercells:
          n_matrix[i,j] = 2*couplconst
        else:
          n_matrix[i,j] = couplconst
  if prnt:
    print(n_matrix)
    eigvals,eigvecs = np.linalg.eig(n_matrix)
    print('eigs =')
    print(eigvals)
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

def centerdif(data,dt): #to differentiate an array
  #using mostly centered differentiation
  difarray = np.empty(len(data))
  #first and last seperately
  difarray[0] = (data[1]-data[0])/dt
  difarray[-1]= (data[-1]-data[-2])/dt
  #finding all the other diffs
  for i in range(1,len(data)-1):
    difarray[i]=(data[i+1]-data[i-1])/(2*dt)
  return difarray

def df(phasevect, couplmatrix, couplcoeff, drivevect):
  dphase = couplcoeff*(np.mod(-np.matmul(couplmatrix,phasevect)+np.pi,np.pi*2)-np.pi)+drivevect #calculating derivative
  return dphase

def RK4(phasevect,dt, couplmatrix,couplcoeff,drivevect):
  K1 = dt* df(phasevect,couplmatrix,couplcoeff,drivevect)
  K2 = dt* df(phasevect+K1/2,couplmatrix,couplcoeff,drivevect)
  K3 = dt* df(phasevect+K2/2,couplmatrix,couplcoeff,drivevect)
  K4 = dt* df(phasevect+K3,couplmatrix,couplcoeff,drivevect)
  newvect = phasevect+K1/6+K2/3+K3/3+K4/6
  return newvect

def couplinglines(ax,coordarray,couplmatrix,num): #draws lines between points
  for i in range(num):
    for j in range(num):
      if couplmatrix[i,j] != 0:
        x_vals = [coordarray[i,0],coordarray[j,0]]
        y_vals = [coordarray[i,1],coordarray[j,1]]
        ax.plot(x_vals,y_vals,c = 'k')
  return ax

def makeframe(i,cellnr,locs,dt,step,couplmatrix,phasearray,
              width,height,dpi,cmap):
  fig = plt.figure(figsize=(width/dpi,height/dpi), dpi = dpi)
  ax = fig.add_subplot(1,1,1)
  plt.xlim((-3.5,3.5))
  plt.ylim((-3.5,3.5))
  for loc in range(cellnr): #we make a cirlce from the locs array
    color = cmap((1+np.sin(phasearray[int(i/(dt*step)),loc]))*0.5) #getting the color for this circle
    circle = plt.Circle(locs[loc,:],0.1,color=color) #making a circle
    ax.add_artist(circle)
  ax = couplinglines(ax, locs,couplmatrix, cellnr)
  vp.despine(ax, all = True)
  # put pixel buffer in numpy array
  canvas = FigureCanvas(fig)
  canvas.draw()
  mat = np.frombuffer(canvas.tostring_rgb(), dtype = 'uint8').reshape(height,width,3)
  # return the matrix and close plots
  plt.close()
  #print('frame {} done, returning'.format(i))
  return mat

def frame_queuer(queue_in,queue_out,frames,cellnr,locs,dt,step,
                 couplmatrix,phasearray,width,height,dpi,cmap):
  while True:
    try:
      framenum = queue_in.get(timeout = 1) #get a job ready
    except queue.Empty:
      break
    else:
      frame = (makeframe(framenum,cellnr,locs,dt,step,couplmatrix,phasearray,width,height,dpi,cmap))
      while True: #coordinating putting the frames on queue_out
        try:
          last = frames.get(timeout = 5)
        except:
          pass
        else:
          if last == framenum-1:
            queue_out.put(frame)
            queue_out.join()
            frames.put(framenum)
            break
          else:
            frames.put(last)
  #print('queuer closing')
  return True

def parallelvideo(cellnr, levels, tEnd,dt, phasearray,couplmatrix,current_time, scalefactor = 0.8, step = 10, width = 1080, height = 1080, dpi = 100):
  print('finding coordinates...')
  locs =  coordhelper(levels, scalefactor)#dividing a circle into even intervals
  print('coords done')
  cmap = mpl.cm.get_cmap('seismic') #color map
  # create OpenCV video writer
  video = cv2.VideoWriter('outputs/recursive/{}.avi'.format(current_time), cv2.VideoWriter_fourcc(*'MP42'), 30, (width,height))
 
  #start parallelization
  framequeue = mp.Queue()#empty queue
  mats = mp.JoinableQueue() #for outputs
  frames = mp.Queue() #to keep track of what frame we're at
  frames.put(-1)#the first frame will be 0 and will check for framenum -1
  processlist = [] #so we can wait for everything to finish
  maxprocesses = 12 #maybe increase this later?
  bar2 = Bar('creating video', max= tEnd*step)
  for i in range(tEnd*step): #creating a queue
    framequeue.put(i)

 # generating frames
  for i in range(maxprocesses):
    #we create a process which calls the frame queuer, these all go through the
    #queue and call the framemaker, which makes it so we can have multiple
    #framemakers going
    p = mp.Process(target=frame_queuer,args=(framequeue,mats,frames,cellnr,locs,dt,step,couplmatrix,phasearray,
            width,height,dpi,cmap,))
    processlist.append(p)
    p.start()

  #we loop untill we've added as many frames as needed
  framesadded = 0
  while framesadded < tEnd*step:
    try:
      newframe = mats.get(timeout = 1)
    except:
      pass
    else:
      mats.task_done()
      video.write(newframe)
      framesadded +=1
      bar2.next()


  while not frames.empty():
    frames.get() #flushing to allow the processes to close
  #print('mats is empty {}'.format(mats.empty()))
  #print('framequeue is empty {}'.format(framequeue.empty()))
  for p in processlist:#shut down all the processes
    p.join()
    p.close()
  
  # close video writer
  bar2.finish()
  cv2.destroyAllWindows()
  video.release()

def videomaker(cellnr, levels, tEnd,dt, phasearray,couplmatrix,current_time, scalefactor = 0.8, step = 10, width = 1080, height = 1080, dpi = 100):
  locs =  coordhelper(levels, scalefactor)#dividing a circle into even intervals
  cmap = mpl.cm.get_cmap('seismic') #color map
  # create OpenCV video writer
  video = cv2.VideoWriter('outputs/recursive/{}.avi'.format(current_time), cv2.VideoWriter_fourcc(*'MP42'), 30, (width,height))
  bar2 = Bar('making video', max = tEnd*step)
  # loop over your images
  for i in range(tEnd*step):
    mat = makeframe(i,cellnr,locs,dt,step,
                    couplmatrix,phasearray,width,height,dpi,cmap)

    # write frame to video
    video.write(mat)
    bar2.next()

  # close video writer
  bar2.finish()
  cv2.destroyAllWindows()
  video.release()

def recursivecell(levels = 3, init = None,tEnd = 100,dt = 0.001,couplcoeff =
                  0.3,drive = 0.5, plots = True,plots2= True, parallel = True, #for if we want the video paralellized or not 
                  vidja = True, scalefactor = 0.8, step = 10, width = 1080, height = 1080, dpi = 100):#video parameters
  #finding amount of cells
  cellnr =cellamount(levels) 
   
  #random startvector
  if np.any(init == None):
    startvect = np.empty((cellnr,1))
    for i in range(cellnr):
      startvect[i,0] = random.random()*2*np.pi
  else:
      startvect = np.array(init)
      startvect = np.reshape(startvect,(cellnr,1))
  initcon = np.array2string(np.reshape(startvect,(1,cellnr)))
  #coupling matrices
  totcoupl = n_matrixgen(levels,0.25)#using function to generate coupling matrix
  drivevect = np.ones(cellnr)*drive
  
  #simulation setup
  phasevect = startvect
  lensarray = np.empty(1)
  lensarray[0] = np.linalg.norm(phasevect)# array of lengths
  t = 0 #+dt
  print(totcoupl)

  #phasearray = np.array([phasevect],) #total array of all the phases
  phasearray = np.empty((cellnr,int(tEnd/dt + 1)))
  phasearray[:,0] = phasevect[:,0]

  bar1 = Bar('simulating', max = int(tEnd/dt))

  stepcounter = 1 #stepcounter since int(t/dt) did strange things
  #actually simulating
  while stepcounter<tEnd/dt+1:
    dphase = couplcoeff*(np.mod(-np.matmul(totcoupl,phasevect)+np.pi,np.pi*2)-np.pi)+drivevect #calculating derivative
    #phasevect = RK4(phasevect,dt,totcoupl,couplcoeff,drivevect)
    phasevect = phasevect+dt*dphase
    lensarray = np.append(lensarray, np.linalg.norm(phasevect))
    phasearray[:,stepcounter] = phasevect[:,0]
    stepcounter += 1
    t = t+dt
    bar1.next()
  bar1.finish()
  
  #finding time for filenames
  now = datetime.now()

  #transposing the array bc I didn't want to rewrite half the plots
  phasearray = np.transpose(phasearray)

  #dumping the whole thing to a csv for later use
  np.savetxt('outputs/recursive/{now}drive{drive}coupl{coup}.csv'.format(now=now,drive=drive,coup=couplcoeff),phasearray)

  if plots:
    #plot phase stuff
    vp.set_palette('cool')
    tarr = np.arange(0,tEnd+1*dt, dt)
    plt.figure(figsize=(15,10))
    for i in range(cellnr):
        plt.plot(tarr, np.mod(phasearray[:,i],2*np.pi))
    plt.savefig('outputs/recursive/phase_{}.png'.format(now))
    plt.close()
    #plot sine
    plt.figure(figsize=(15,5))
    for i in range(cellnr):
       plt.plot(tarr, np.sin(phasearray[:,i]))

    plt.savefig('outputs/recursive/sine_{}.png'.format(now))
    plt.close()
  #plots that are more useful for bigger sytems
  if plots2:
    cool = vp.palette('cool')#let's make em pretty
    plotbar = Bar('plotting...', max = cellnr)
    tarr = np.arange(0,tEnd+1*dt, dt)
    fig = plt.figure(figsize=(15, 5*math.ceil(cellnr/3)))
    for i in range(cellnr):
      data = phasearray[:,i]
      sins = np.sin(data)
      diffs = centerdif(sins,dt)

      #phasespace
      ax = fig.add_subplot(int(math.ceil(cellnr/3)),3,i+1)
      ax.plot(sins,diffs,c=cool[np.mod(i,len(cool))])
      ax.set_xlabel('x{}'.format(i+1))
      ax.set_ylabel("x'{}".format(i+1))
      ax.set_ylim(top = 1, bottom=-1)
      plotbar.next()
    plt.savefig('outputs/recursive/{}phaseplane.png'.format(now))
    plotbar.finish()
    plt.close()
    #sineplots
    fig2 = plt.figure(figsize=(15, 5*math.ceil(cellnr/3)))
    plotbar = Bar('plotting...', max = cellnr)
    for i in range(cellnr):
      data = phasearray[:,i]
      sins = np.sin(data)
      ax2 = fig2.add_subplot(cellnr,1,i+1)
      ax2.plot(tarr,sins,c=cool[np.mod(i,len(cool))])
      ax2.set_xlabel('t')
      ax2.set_ylabel("x{}".format(i+1))
      plotbar.next()
    plt.savefig('outputs/recursive/{}sins.png'.format(now))
    plt.close()
    plotbar.finish()

  #Video stuff
  if vidja == True:
    if parallel == True:
      parallelvideo(cellnr,levels,tEnd,dt, phasearray,totcoupl,now,
               scalefactor,step,width,height,dpi)
    else:
      videomaker(cellnr,levels,tEnd,dt,phasearray,totcoupl,now,scalefactor,step,width,height,dpi)

