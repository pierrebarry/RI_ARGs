import pandas as pd
import tskit as tsk
import numpy as np
from multiprocessing import Pool
from datetime import datetime
import json
import os
import sys

#### -- Usage
# python get_ismonophyletic.py OUTPUT_NAME SPECIES
#

output_json = sys.argv[1]
sp=sys.argv[2]
print(sp)


perFile = True

if (perFile) :
        inputdir = '/shared/projects/cogediv/TREE/tsdate/'+sp+'/'
        inputs = os.popen("ls /shared/projects/cogediv/TREE/tsdate/"+sp+"/").read().split("\n")[:-1]
        #inputs = [ # List of input .trees files
        #       "cgal379.trees",
        #       # "Cgale_87_dated.trees",
        #       # "Cgale_952_dated.trees"
        #]
else :
        inputdir = 'B:\\storage\\general-elias\\Code\\TS_FILES'
        inputs = os.listdir(inputdir) # List of input .trees files

startTime = datetime.now() # For performance measurement

print("Start")

def isRecipMonophyletic(tree, pop_by_node, pop_groups) :
  nodes_by_pop = pd.DataFrame(columns=['nodes']) # This will be the list of leaves per population
  for root in tree.roots:
    for leaf in tree.leaves(root):
      current_pop = pop_by_node.loc[leaf, 'pop'] # Get the current leaf's population
      if (current_pop not in nodes_by_pop.index):
        nodes_by_pop.loc[current_pop] = [set()] # Create a new entry for the population
      nodes_by_pop.at[current_pop, 'nodes'].add(leaf) # Add the leaf to the population
  for pop_group in pop_groups:
    nodes_by_pop.loc[pop_group[0], 'nodes'] = nodes_by_pop.loc[pop_group[0], 'nodes'].union(nodes_by_pop.loc[pop_group[1], 'nodes']) # Combine the two populations according to >
    nodes_by_pop.drop(pop_group[1], inplace=True)

  nodes_by_pop['mrca'] = nodes_by_pop['nodes'].apply(lambda x: tree.mrca(*x)) # Identify the MRCA of the population's leaves
  nodes_by_pop['toplevelChildren'] = nodes_by_pop['mrca'].apply(lambda x: set(tree.leaves(x))) # Identify the MRCA's children
  return (nodes_by_pop['nodes'] == nodes_by_pop['toplevelChildren']).all()

def loadFile(filename) : # Does all the necessary imports on a single file
  sqObj = tsk.load(inputdir + filename)
  populations = str(sqObj.tables.populations.asdict()) # For later check, see cell 4
  pop_by_node = pd.DataFrame({ # Only one of these will be kept, thus the check above
      "pop": [sqObj.tables.nodes[leaf].population for leaf in sqObj.samples()], # Get the population of each leaf
  })
  pd_sequence = pd.DataFrame(
      {
          'file': filename, # Gets the originating filename for future reference
          'span': [tree.span for tree in sqObj.trees()],
          'bounds': [(tree.interval.left, tree.interval.right) for tree in sqObj.trees()], # Loads the tree's bound in a tuple
          'treeObj' : sqObj.aslist()
      },
      index=[tree.index for tree in sqObj.trees()]
  )
  return sqObj, populations, pop_by_node, pd_sequence

#inputdir = "/shared/projects/cogediv/TREE/tsdate/Cgale/"
#inputs = ["Cgale_181406_dated.trees"]

files = pd.DataFrame({ # Initiates the DataFrame with the input files
    'file': inputs,
})

# Fill the files DataFrame with the necessary information
files[['sqObj', 'populations', 'pop_by_node', 'pd_sequence']] = files.apply(lambda row: loadFile(row['file']), axis=1, result_type="expand")

# *The* check I was talking about earlier. If the populations are not the same, that's a problem...
if files.populations.nunique() != 1:
  raise ValueError("All the files do not have the same populations. Are you sure this is the same species ?")
else :
  pop_by_node = files.pop_by_node[0]

pop_groups = [ # Group the populations according to real data
    [0, 1],
    [2, 3]
]

# Define a list of predefined colors
predefined_colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "pink"] # Maximum 8 populations...

# Initialize node_colours dictionary
node_colours = {}

# Example :
exSq = files.iloc[0].sqObj

for node_index, node in enumerate(exSq.tables.nodes):
  if (node.flags & tsk.NODE_IS_SAMPLE) != 0: # If node is a sample

    # Assign color from predefined list, cycling through colors if necessary
    color = predefined_colors[node.population % len(predefined_colors)]
    node_colours[node_index] = color

globalTS = pd.concat(files.pd_sequence.to_list(), ignore_index=True)

n_threads = 10 # Threads to use for multiprocessing, 0 means no multiprocessing
n_samples = 100 # 0 means no bootstrapping

def f(i) : # Define util to be run in parallel
  print(i)
  np.random.seed(i + np.random.randint(0, 10000)) # Necessary, because otherwise all workers will have the same seed
  bootstrap = globalTS.sample(n=int(len(globalTS) * 0.9), replace=True) # Use pandas sample method to take random trees in the sequence for monophyly test
  bootstrap['monophyletic'] = bootstrap.apply(lambda x: isRecipMonophyletic(x.treeObj, pop_by_node, pop_groups), axis=1) # Runs the test
  return bootstrap[bootstrap['monophyletic'] == True]['span'].sum() / bootstrap['span'].sum() # Gets the percentage of the sample trees that is monophyletic

print("Before")
if n_threads > 1 :
  pool = Pool(n_threads) # Create a pool of 2 workers
  percentages = np.array(pool.map(f, range(n_samples))) * 100 # Run bootstrap twice in parallel
else :
  percentages = np.array(list(map(f, range(n_samples)))) * 100

print("Run")
globalTS['monophyletic'] = globalTS.apply(lambda x: isRecipMonophyletic(x.treeObj, pop_by_node, pop_groups), axis=1)
singlepercentage = globalTS[globalTS['monophyletic'] == True]['span'].sum() / globalTS['span'].sum() * 100

if n_samples > 0 :
  sampling = {
  "meanSamplesPercentage": percentages.mean(),
  "samplesStdDev": percentages.std(),
  "confidence_interval": {
        "bounds": [5, 95],
        "lower": np.percentile(percentages, 5),
        "upper": np.percentile(percentages, 95)
  },
  }
else :
  sampling = {}

endTime = datetime.now()

output = {
    "input": {
        "dir": inputdir,
        "files": inputs
        },
    "test": "reciprocal_monophyly",
    "description": "Reciprocal monophyly test on the tree sequence",
    "analysis_settings": {
        "bootstrap_samples": n_samples,
        "pop_groups": pop_groups,
    },
    "perf" : {
        "multiprocessing": n_threads > 1,
        "threads": n_threads,
        "start": startTime,
        "end": endTime,
        "duration": endTime - startTime
    },
    "result": {
        "sampling": sampling,
        "raw": {
                        "positive": int((globalTS['monophyletic'] == True).sum()),
                        "negative": int((globalTS['monophyletic'] == False).sum()),
                        "total_trees": len(globalTS),
                        "percentage": singlepercentage,
                        "details": globalTS[['file', 'bounds', 'span', 'monophyletic']].to_dict(orient='records')
                },
        }
}

json.dump(output, default=str, indent=4, fp=open(output_json, "w"))

print("Over")
