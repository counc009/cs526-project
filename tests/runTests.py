import glob
import subprocess
from tqdm import tqdm
import sys

reps = 1000
repsValgrind = 10
argN = '1000'
argN2 = '50'
argM = '50'
argSeed = '314159'

executables = glob.glob('./*/*.exec')
references = map(lambda nm: nm.replace('.exec', '.ref'), executables)

for ex, ref in zip(executables, references):
  # Skip large tests and Olden
  if ex.startswith('./large_tests') or ex.startswith('./olden'):
    continue

  args = [argN, argSeed]
  # Add extra argument for the nested tests
  # Also use smaller n to keep runtimes reasonable
  if ex.startswith('./nested'):
    args = [argN2, argM, argSeed]

  print('Testing', ex)
  refProc = subprocess.Popen([ref] + args, stdout=subprocess.PIPE)
  refResult = refProc.communicate()[0]
  if refProc.returncode != 0:
    print('Reference returned non-0, exiting')
    sys.exit(1)

  for i in tqdm(range(reps)):
    execProc = subprocess.Popen([ex] + args, stdout=subprocess.PIPE)
    execResult = execProc.communicate()[0]
    if execProc.returncode != 0:
      print('Executable returned non-0, failed')
      sys.exit(1)
    if execResult != refResult:
      print('Executable produced different output, failed')
      sys.exit(1)
  for i in tqdm(range(repsValgrind)):
    # Flags cause valgrind to return 1 if there are any leaks
    execProc = subprocess.Popen(['valgrind', '--error-exitcode=1',
                                 '--leak-check=full', ex] + args,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
    execResult = execProc.communicate()[0]
    if execProc.returncode != 0:
      print('Executable returned non-0, failed')
      sys.exit(1)
    if execResult != refResult:
      print('Executable produced different output, failed')
      sys.exit(1)
