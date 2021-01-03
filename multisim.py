from recursive_chaos import *
import multiprocessing as mp
import queue
import numpy as np

def simcaller(ins):
  drive = ins[0]
  coupling = ins[1]
  recursivecell(tEnd = 200, couplcoeff=coupling, drive=drive,
                plots2=False,parallel=False)
  return

def simqueue():
  while True:
    try:
      ins = queue.get(timeout=3)
    except queue.Empty:
      break
    else:
      simcaller(ins)
  return

if __name__ == '__main__':
  #defining some variables
  maxprocesses = 12
  drivemin = 0.1
  couplingmin = 0
  drivemax = 1.5
  couplingmax = 1
  step = 0.1

  #empty list for processes
  processes = []

  #set up a queue
  queue = mp.Queue()
  #first generate the inputs, for each drive and coupling we want it running 3 times
  for i in np.arange(drivemin, drivemax+step,step):
    for j in np.arange(couplingmin, couplingmax+step, step):
      for k in range(3):
        queue.put((i,j))
  
  #just printing how many we're doing
  simnum = ((drivemax-drivemin)/step)*((couplingmax-couplingmin)/step)*3
  print('starting {} simulations!'.format(simnum))

  #start the processes
  for i in range(maxprocesses):
    p = mp.Process(target= simqueue,)
    processes.append(p)
    p.start()

  #wait for the processes to finish and close them
  for p in processes:
    p.join()
    p.close()
