import glob
import subprocess
from tqdm import tqdm
import sys

reps = 100
argN = '100'
argSeed = '314159'

executables = glob.glob('./*/*.exec')
references = map(lambda nm: nm.replace('.exec', '.ref'), executables)

for ex, ref in zip(executables, references):
  print('Testing', ex)
  refProc = subprocess.Popen([ref, argN, argSeed], stdout=subprocess.PIPE)
  refResult = refProc.communicate()[0]
  if refProc.returncode != 0:
    print('Reference returned non-0, exiting')
    sys.exit(1)

  for i in tqdm(range(reps)):
    execProc = subprocess.Popen([ex, argN, argSeed], stdout=subprocess.PIPE)
    execResult = execProc.communicate()[0]
    if execProc.returncode != 0:
      print('Executable returned non-0, failed')
      sys.exit(1)
    if execResult != refResult:
      print('Executable produced different output, failed')
      sys.exit(1)
