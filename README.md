# HW6-HMM

In this assignment, you'll implement the Forward and Viterbi Algorithms (dynamic programming). 

[![Build Status](https://github.com/myers2004/HW6-HMM/actions/workflows/test.yml/badge.svg?event=push)](https://github.com/myers2004/HW6-HMM/actions/workflows/test.yml)


# Assignment

## Overview 

The goal of this assignment is to implement the Forward and Viterbi Algorithms for Hidden Markov Models (HMMs).

For a helpful refresher on HMMs and the Forward and Viterbi Algorithms you can check out the resources [here](https://web.stanford.edu/~jurafsky/slp3/A.pdf), 
[here](https://towardsdatascience.com/markov-and-hidden-markov-model-3eec42298d75), and [here](https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/). 





## Tasks and Data 
Please complete the `forward` and `viterbi` functions in the HiddenMarkovModel class. 

We have provided two HMM models (mini_weather_hmm.npz and full_weather_hmm.npz) which explore the relationships between observable weather phenomenon and the temperature outside. Start with the mini_weather_hmm model for testing and debugging. Both include the following arrays:
* `hidden_states`: list of possible hidden states 
* `observation_states`: list of possible observation states 
* `prior_p`: prior probabilities of hidden states (in order given in `hidden_states`) 
* `transition_p`: transition probabilities of hidden states (in order given in `hidden_states`)
* `emission_p`: emission probabilities (`hidden_states` --> `observation_states`)



For both datasets, we also provide input observation sequences and the solution for their best hidden state sequences. 
 * `observation_state_sequence`: observation sequence to test 
* `best_hidden_state_sequence`: correct viterbi hidden state sequence 


Create an HMM class instance for both models and test that your Forward and Viterbi implementation returns the correct probabilities and hidden state sequence for each of the observation sequences.

Within your code, consider the scope of the inputs and how the different parameters of the input data could break the bounds of your implementation.
  * Do your model probabilites add up to the correct values? Is scaling required?
  * How will your model handle zero-probability transitions? 
  * Are the inputs in compatible shapes/sizes which each other? 
  * Any other edge cases you can think of?
  * Ensure that your code accomodates at least 2 possible edge cases. 

Finally, please update your README with a brief description of your methods.


## Methods

### Foward Algorithm

The foward algorithm computes the probability of a given observation sequence occuring given the priors, the transistion probabilites, and the emission probabilites for the hidden states in a Hidden Markov Model. A brief step-by-step description follows:

Step 1) Create a matrix with n rows and m columns, where n is the number of hidden states, and m is the number of observations in the sequence. Each column corresponds to an observation in the sequence.

Step 2) Intilize the first column of the matrix by mutliplying the prior of each hidden state by the emission probability of that state for the first observation.

Step 3) To fill in the probability for the next hidden state, iterate over the probablitles of each hidden state for the previous observation, summing the products of the last probability, the transition pribability from the corresponding hidden state to the current, and the emission proabililty of the current observation from the curren hidden state. Repeat this to fill in the whole matrix one column at a time.

Step 4) Sum the probabilies in the last column of the matrix to get the probability of the given observation sequence.

### Viterbi Algorithm

The Viterbi algorithm computes the most likely hidden state sequence for a given sequence of observations of a Hidden Mqrkov Model. It once again needs the priors, the transistion probabilites, and the emission probabilites for the hidden states. A brief step-by-step description follows:

Step 1)  Create a matrix called a Viterbi table with n rows and m columns, where n is the number of hidden states, and m is the number of observations in the sequence. Each column corresponds to an observation in the sequence, and each row corresponds to a hidden state.

Step 2) Create another matrix with n row and m columns which will be used to store the path taken through the Viterbi table for back tracing.

Step 3) Calculate the first column probabilites by multiplying the prior of each hidden state by the emission probability of the first observation in the sequence

Step 4) Calculate all the rest of the probabilites one-at-a-time. To do the first, move to the first row in the next (the second) column of the Viterbi table. For each entry in the previous column (the first), multiply the entry by the transition probability from the entries corresponding hidden state to the curent hidden state. Take the max of these products, and store with row this max came from in the back tracing matrix. Then multiply the max by the emission probability of the current hidden state to the current observation in the sequence. Now move down to the next row in the current column and repeat. Once this column is filled, move to the next column and repeat. Do this until the Viterbi tabkle and backtracing matrix are filled. 

Step 5) Intilize a list to hold the indices of the best hidden sequence.

Step 6) Go through the last column and select the entry with the highest probability. Add the corresponding hidden state index to the best hidden sequence list.

Step 7) Follow the back trace matrix to the next hidden state. Add the corresponding index to the best hidden sequence list. Repeat until we get to the first column of the Viterbi table.

Step 8) Covert the best seqeunce indices to the best hidden sequence, and reverse the list.

## Task List

[TODO] Complete the HiddenMarkovModel Class methods  <br>
  [X] complete the `forward` function in the HiddenMarkovModelClass <br>
  [X] complete the `viterbi` function in the HiddenMarkovModelClass <br>

[TODO] Unit Testing  <br>
  [X] Ensure functionality on mini and full weather dataset <br>
  [X] Account for edge cases 

[TODO] Packaging <br>
  [X] Update README with description of your methods <br>
  [X] pip installable module (optional)<br>
  [X] github actions (install + pytest) (optional)


## Completing the Assignment 
Push your code to GitHub with passing unit tests, and submit a link to your repository [here](https://forms.gle/xw98ZVQjaJvZaAzSA)

### Grading 

* Algorithm implementation (6 points)
    * Forward algorithm is correct (2)
    * Viterbi is correct (2)
    * Output is correct on small weather dataset (1)
    * Output is correct on full weather dataset (1)

* Unit Tests (3 points)
    * Mini model unit test (1)
    * Full model unit test (1)
    * Edge cases (1)

* Style (1 point)
    * Readable code and updated README with a description of your methods 

* Extra credit (0.5 points)
    * Pip installable and Github actions (0.5)
