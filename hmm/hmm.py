import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p
        
        ##Error Handling

        num_priors = np.shape(self.prior_p)[0]
        num_hidden = len(self.hidden_states)
        num_os = len(self.observation_states)

        epsilon = 0.0000001

        #Handling imput matrices of wrong dimension
        if num_priors != num_hidden:
            raise(IndexError('Number of prior probabilites does not match number of hideen states'))
        
        if np.shape(self.transition_p)[0] != num_hidden or np.shape(self.transition_p)[1] != num_hidden:
            raise(IndexError('Dimensions of transition probability matrix do not match number of hidden states'))
        
        if np.shape(self.emission_p)[0] != num_hidden:
            raise(IndexError('Number of columns of emission probability matrix do not match number of hidden states'))

        if np.shape(self.emission_p)[1] != num_os:
            raise(IndexError('Number of rows of emission probability matrix do not match number of observation states'))
        
        #Ensurng probabilites for each hidden state add to one

        sum = 0
        for i in range(num_priors):
            sum+= self.prior_p[i]
        if not np.isclose(sum,1,epsilon):
            raise(ValueError('Prior probabilites don\'t sum to 1'))

        for i in range(num_hidden):
            sum = 0
            for j in range(num_os):
                sum += self.emission_p[i][j]
            if not np.isclose(sum,1,epsilon):
                raise(ValueError('Emission probabilites for a hidden state don\'t sum to 1'))

        for i in range(num_hidden):
            sum = 0
            for j in range(num_hidden):
                sum += self.transition_p[i][j]
            if not np.isclose(sum,1,epsilon):
                raise(ValueError('Transistion probabilites for a hidden state don\'t sum to 1'))




    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        num_obs = len(input_observation_states)
        num_states = len(self.hidden_states)

        alpha_mat = np.empty((num_states, num_obs)) #will hold all of the foward algorithm probabilities

       
        # Step 2. Calculate probabilities

        #init alpha_1(j)s
        init_obs_index = self.observation_states_dict[input_observation_states[0]]
        for i in range(num_states):
            alpha_mat[i][0] = self.prior_p[i]*self.emission_p[i][init_obs_index]
        
        #then all the rest
        for j in range(1,num_obs):
            for i in range(num_states):
                sum = 0
                obs_index = self.observation_states_dict[input_observation_states[j]]
                for k in range(num_states):
                    sum += alpha_mat[k][j-1]*self.transition_p[k][i]*self.emission_p[i][obs_index]
                alpha_mat[i][j] = sum

        # Step 3. Return final probability
         
        #sum final probabilites
        final_prob = 0
        for i in range(num_states):
            final_prob += alpha_mat[i][num_obs-1]
        return final_prob
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """        
        
        # Step 1. Initialize variables
        num_obs = len(decode_observation_states)
        num_states = len(self.hidden_states)
        
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((num_states,num_obs))

        #store the prev enrty for traceback
        paths_mat = np.zeros((num_states,num_obs))
       
       # Step 2. Calculate Probabilities

       #init first 
        init_obs_index = self.observation_states_dict[decode_observation_states[0]]
        for i in range(num_states):
            viterbi_table[i][0] = self.prior_p[i]*self.emission_p[i][init_obs_index]

        #then all the rest
        for j in range(1,num_obs):
            for i in range(num_states):
                obs_index = self.observation_states_dict[decode_observation_states[j]]
                max_prev = -np.inf
                for k in range(num_states):
                    curr_prev = viterbi_table[k][j-1]*self.transition_p[k][i]
                    if curr_prev > max_prev:
                        max_prev = curr_prev
                        paths_mat[i][j] = k
                viterbi_table[i][j] = self.emission_p[i][obs_index] * max_prev

            
        # Step 3. Traceback
        best_path_index = []

        next_state_index = 0

        print(viterbi_table)
        print(paths_mat)
        max_prob = -np.inf
        for i in range(num_states):
            curr_prob = viterbi_table[i][num_obs-1]
            if curr_prob > max_prob:
                max_prob = curr_prob
                next_state_index = i

        best_path_index.append(next_state_index)
        j = num_obs - 1
        while j > 0:
            next_state_index = int(paths_mat[next_state_index][j])
            best_path_index.append(next_state_index)
            j = j - 1

        #convert index to states and reverse the best path
        best_path = []
        for i in range(num_obs):
            best_path.append(self.hidden_states_dict[best_path_index[-i-1]])

        # Step 4. Return best hidden state sequence 
        
        return best_path