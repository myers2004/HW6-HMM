import pytest
from hmm import HiddenMarkovModel
import numpy as np




def test_mini_weather():
    """
    TODO: 
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm=np.load('./data/mini_weather_hmm.npz')
    mini_input=np.load('./data/mini_weather_sequences.npz')

    epsilon = 0.000000000001

    #Extract variables from .npz
    hs = mini_hmm[mini_hmm.files[0]]
    os = mini_hmm[mini_hmm.files[1]]
    prior = mini_hmm[mini_hmm.files[2]]
    tp = mini_hmm[mini_hmm.files[3]]
    ep = mini_hmm[mini_hmm.files[4]]

    hs_best = mini_input[mini_input.files[1]]
    os_seq = mini_input[mini_input.files[0]]

    test_mini = HiddenMarkovModel(os,hs,prior,tp,ep)

    ##Testing foward method

    correct_final_foward = 0.035064411621

    hmm_foward_prob = test_mini.forward(os_seq)

    assert np.isclose(hmm_foward_prob, correct_final_foward, epsilon)

    ##Testing viterbi method

    hmm_best_path = test_mini.viterbi(os_seq)

    #Assert the length of the calculated best path is the same 
    # as the observation state sequence
    assert len(hmm_best_path) == len(os_seq)

    #Assert the correct hidden state path is found
    assert len(hmm_best_path) == len(hs_best)
    for i in range(len(hmm_best_path)):
        assert hmm_best_path[i] == hs_best[i]

    ##Testing if nonsense input is handled correctly

    #Test if an error is raised with the wrong number of priors
    prior_wrong_size = np.array((0.25, 0.5, 0, 0.25))
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior_wrong_size,tp,ep)

    #Test if an error is raised if priors don't add to 1
    prior_not_sum_1 = np.array((0.5,0.7))
    with pytest.raises(ValueError):
        HiddenMarkovModel(os,hs,prior_not_sum_1,tp,ep)

    #Test if an error is raised if a transition matrix of wrong dimensions is input
    transition_wrong_size = np.array([[0.2,0.15],[0.8,0.35],[0.0,0.5]])
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior,transition_wrong_size,ep)


    #Test if an error is raised if transistion probabilites for each hidden state don't add to 1
    transition_not_sum_zero = np.array([[0.3,0.75],[0.8,0.35]])
    with pytest.raises(ValueError):
        HiddenMarkovModel(os,hs,prior,transition_not_sum_zero,ep)
    
    #Test if an error is raised if an emission matrix with the wrong number of rows is input
    ep_wrong_num_rows = np.array([[0.2,0.15],[0.8,0.35],[0.0,0.5]])
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior,tp,ep_wrong_num_rows)
    
    #Test if an error is raised if an emission matrix with the wrong number of rcolumns is input
    ep_wrong_num_col = np.array([[0.3,0.75,0.45],[0.7,0.25,0.55]])
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior,tp,ep_wrong_num_col)
    
    #Test if an error is raised if the emission probability for a hidden state doesn;t equal 1
    ep_not_sum_zero = np.array([[0.9,0.65],[0.15,0.35]])
    with pytest.raises(ValueError):
        HiddenMarkovModel(os,hs,prior,tp,ep_not_sum_zero)



def test_full_weather():

    """
    TODO: 
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    full_hmm=np.load('./data/full_weather_hmm.npz')
    full_input=np.load('./data/full_weather_sequences.npz')

    #Extract variables from .npz
    hs = full_hmm[full_hmm.files[0]]
    os = full_hmm[full_hmm.files[1]]
    prior = full_hmm[full_hmm.files[2]]
    tp = full_hmm[full_hmm.files[3]]
    ep = full_hmm[full_hmm.files[4]]

    hs_best = full_input[full_input.files[1]]
    os_seq = full_input[full_input.files[0]]

    test_mini = HiddenMarkovModel(os,hs,prior,tp,ep)
   
    hmm_best_path = test_mini.viterbi(os_seq)
    #Assert the length of the calculated best path is the same 
    # as the observation state sequence
    assert len(hmm_best_path) == len(os_seq)

    #Assert the correct hidden state path is found
    assert len(hmm_best_path) == len(hs_best)
    for i in range(len(hmm_best_path)):
        assert hmm_best_path[i] == hs_best[i]

    ##Testing if nonsense input is handled correctly

    #Note that dimesion error handling is done before probablility summing to 1
    # -> Using np.zeros to test dimension mismatch

    num_obs = len(os)
    num_hs = len(hs)

    #Test if an error is raised with the wrong number of priors
    prior_wrong_size = np.zeros((num_hs + 1))
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior_wrong_size,tp,ep)

    #Test if an error is raised if a transition matrix of wrong dimensions is input
    transition_wrong_size = np.zeros((num_hs + 1, num_hs - 1))
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior,transition_wrong_size,ep)

    
    #Test if an error is raised if an emission matrix with the wrong number of rows is input
    ep_wrong_num_rows = np.zeros((num_hs + 1, num_obs))
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior,tp,ep_wrong_num_rows)
    
    #Test if an error is raised if an emission matrix with the wrong number of rcolumns is input
    ep_wrong_num_col = np.zeros((num_hs, num_obs + 1))
    with pytest.raises(IndexError):
        HiddenMarkovModel(os,hs,prior,tp,ep_wrong_num_col)













