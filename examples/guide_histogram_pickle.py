#[ guide_histogram_pickle

from __future__ import print_function
import histogram as hg
import pickle

h1 = hg.histogram(hg.axis.regular(2, -1, 1))
h2 = hg.histogram(h1) # creates copy
h1(-0.5)
h2(0.5)

# arithmetic operators (see performance note below)
h3 = h1 + h2
h4 = h3 * 2

print(h4.bin(0).value, h4.bin(1).value)
# prints: 2.0 2.0

# now save the histogram
with open("h4_saved.pkl", "w") as f:
    pickle.dump(h4, f)
with open("h4_saved.pkl", "r") as f:
    h5 = pickle.load(f)

print(h4 == h5)
# prints: True

#]