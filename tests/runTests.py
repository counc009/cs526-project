import glob
import subprocess
from tqdm import tqdm
import sys
import time

reps = 1000
repsValgrind = 10
argN = '1000'
argSeed = '314159'

executables = glob.glob('./*/*.exec')
references = map(lambda nm: nm.replace('.exec', '.ref'), executables)

for ex, ref in zip(executables, references):
  print('Testing', ex)
  #start_time = time.time()
  for i in tqdm(range(reps)):
    refProc = subprocess.Popen([ref, argN, argSeed], stdout=subprocess.PIPE)
    refResult = refProc.communicate()[0]
		#print("Reference Run %s seconds ---" % (time.time() - start_time))
    if refProc.returncode != 0:
      print('Reference returned non-0, exiting')
      sys.exit(1)

  #start_time = time.time()
  for i in tqdm(range(reps)):
    execProc = subprocess.Popen([ex, argN, argSeed], stdout=subprocess.PIPE)
    execResult = execProc.communicate()[0]
    if execProc.returncode != 0:
      print('Executable returned non-0, failed')
      sys.exit(1)
    if execResult != refResult:
      print('Executable produced different output, failed')
      sys.exit(1)
  #elapsed = time.time() - start_time
  #print("Exec Run for all reps %s seconds ---" % (elapsed))
  #print("Average time taken ---- %s seconds " % (elapsed/reps) )
  for i in tqdm(range(repsValgrind)):
    execProc = subprocess.Popen(['valgrind', ex, argN, argSeed],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    execResult = execProc.communicate()[0]
    if execProc.returncode != 0:
      print('Executable returned non-0, failed')
      sys.exit(1)
    if execResult != refResult:
      print('Executable produced different output, failed')
      sys.exit(1)
