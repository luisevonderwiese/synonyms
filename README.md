Experiments for the publication "Playing the Joker for Cognate Data - Approaches for Integrating out Subjectivity in Cognate Synonym Selection"

## Requirements
* Python 3 including additional packages:
  + scipy
  + tabulate
  + matplotlib
* [PyPythia](https://github.com/tschuelia/PyPythia/) version 1.1.4
* [lingdata](https://github.com/luisevonderwiese/lingdata) version ?

Precompiled binaries for [RAxML-NG](https://github.com/amkozlov/raxml-ng) version 1.2.0 and [qdist](https://birc.au.dk/software/qdist) are contained in `bin/`.

## Execution
```
python experiment.py
```
(Executes tree inferences with RAxML-NG, calculates distances of resulting trees and determine difficulty scores with Pythia, may be executed on a remote machine)
```
python analysis.py
```
(Evaluates results, may be executed locally after copying `data/results/`)
