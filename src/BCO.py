from locale import normalize
from utils import *
import time
import robosuite as suite
from robosuite import load_controller_config
import h5py 
import random
import matplotlib.pyplot as plt
import math

controller_config = load_controller_config(default_controller='OSC_POSE')
print("hello")
class BCO():
    def __init__(self, state_shape, action_shape):
        # initial values
        self.state_dim = state_shape
        self.action_dim = action_shape
        self.max_episodes = args.max_episodes      # maximum episode
        self.alpha = 0.1                           # alpha = | post_demo | / | pre_demo |
        self.learning_rate = 0.001
        #mins and maxs of actions and states according to expert's data
        
        self.action_mean = [ 3.30868852e-01,  5.01613115e-02, -4.44982295e-01,  5.30026593e-04, 1.56484243e-02,  2.70126153e-02, -9.94754098e-01]
        self.action_std = [0.23106101, 0.12565047, 0.20204806, 0.02310043, 0.06156015, 0.10668751, 0.10229508]
        self.state_mean = [ 6.00000000e-01,  6.89390088e-03,  4.86273448e-01,  8.47384877e-03,
 -2.34883840e+00, -1.55260130e-02,  2.89784988e+00,  7.83876668e-01,
  3.57189713e-02, -3.57044436e-02,  1.39705945e-03,  2.07652219e-02,
  8.20896252e-01, -6.40670677e-02,  5.82981129e-06, -2.38215497e-06,
  5.90432100e-01,  9.32411214e-03,  4.52622481e-01,  1.08494021e-02,
  4.10242683e-01, -2.94357829e-02, -2.52959484e-03, -1.16721624e-02,
  1.48789914e-02, -1.49444430e-02,  1.96993879e-05, -1.48051372e-06,
 -8.53228925e-03, -2.47378877e-04, -4.69794280e-04, -2.20185800e-05]


        self.state_std = [3.60555128e-01, 3.17536597e-02, 2.00009160e-01, 2.70136539e-02,
 1.86435769e-01, 5.17471187e-02, 6.57514374e-02, 1.76936612e-01,
 6.00861791e-03, 6.01102627e-03, 1.57843873e-02, 5.85406290e-03,
 2.21918469e-03, 7.19224314e-01, 1.05618622e-06, 9.97323756e-07,
 3.21042403e-01, 6.05031817e-02, 2.05199377e-01, 7.03129604e-02,
 2.21289572e-01, 1.05656446e-01, 1.76657897e-01, 2.95540062e-01,
 2.51516170e-02, 2.51733845e-02, 4.32103275e-06, 4.38396855e-06,
 5.57054799e-02, 2.21830219e-04, 2.05036859e-04, 3.07063172e-10]

        # initial sessions
        self.sess = tf.compat.v1.Session()

        self.env = suite.make(
            env_name = "Lift",
            robots = "Panda",
            controller_configs = controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )
        
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.disable_eager_execution()

        #input placeholder
        self.state = tf.compat.v1.placeholder(tf.float32, [None, self.state_dim], name="state")
        self.nstate = tf.compat.v1.placeholder(tf.float32, [None, self.state_dim], name="next_state")
        self.action = tf.compat.v1.placeholder(tf.float32, [None, self.action_dim], name="action")

        # set policy model 
        self.policy()
        self.inverse_dynamic_model()
        self.random_policy()

        #tensorboard output for policy loss, idm loss and reward
        self.writer = tf.compat.v1.summary.FileWriter('logdir2/', self.sess.graph)
        self.plc_loss_writer = tf.compat.v1.summary.FileWriter('logdir_rlc-64/plc_loss')
        self.idm_loss_writer = tf.compat.v1.summary.FileWriter('logdir_rlc-64/idm_loss')
        self.reward_writer = tf.compat.v1.summary.FileWriter('logdir_rlc-64/reward_loss')
        # test time
        self.test_time= 100


    def load_demonstrations(self):
        """ 
        Load demostrations from the robomimic dataset from txt 
        """ 
        states = []
        nstates = []
        actions = []

        for state in open(args.state_dataset):
            s = state.replace("[", "").replace("\n", "").replace("]","").split()
            s_values = [float(x) for x in s]
            states.append(s_values)
        for nstate in open(args.nstate_dataset):
            ns = nstate.replace("[", "").replace("\n", "").replace("]","").split()
            ns_values = [float(x) for x in ns]
            nstates.append(ns_values)
        for action in open(args.action_dataset):
            a = action.replace("[", "").replace("\n", "").replace("]","").split()
            a_values = [float(x) for x in a]
            a_values.pop(6)
            actions.append(a_values)
        num_states = len(states)
        num_nstates = len(nstates)
        num_actions = len(actions)
        

        return num_states, num_nstates, num_actions, states, nstates, actions
    
    def D_demo(self, states, nstates):
        """
        State and next state normalization
        """

        norm_S = []
        norm_nS = []
        for state in states:
            norm_state = self.state_normalization(self.state_mean, self.state_std, state)
            norm_S.append(norm_state) 
        for nstate in nstates:
            norm_nstate = self.state_normalization(self.state_mean, self.state_std, nstate)
            norm_nS.append(norm_nstate)

        return norm_S, norm_nS

    def T_demo(self):
        """
        Generate a set of states and next states from D_demo 
        """
        
        sample_idx = range(self.num_states)
        sample_index = np.random.choice(sample_idx, int(round(self.num_sample)))
        S = [self.norm_S[i] for i in sample_index]
        nS = [self.norm_nS[i] for i in sample_index]
       
        lS = len(S)
        lnS = len(nS)

        return S, nS


    def logit_normalization(self, x_min_arr, x_max_arr, logits):
        """
        Normalize each feature of each logit with z-score normalization 
        """
        x_mean = tf.constant(x_min_arr, dtype=logits.dtype)
        x_std = tf.constant(x_max_arr, dtype=logits.dtype)

        norm_logits = (logits - x_mean) / x_std
        
        return norm_logits

    def action_unormalization(self, a_min, a_max, plc_action):
        """
        Unormalize the action
        """

        unormalized_action= []
        for i in range(len(plc_action)):
            x_mean = a_min[i]
            x_std = a_max[i]
            feature = plc_action[i]
            un_feature = (feature * x_std) + x_mean
            unormalized_action.append(un_feature)
        
        return unormalized_action
    def random_policy(self):
        """ 
        Create the policy model 
        """
        
        rplc_input = self.state
       
        rplc_dense1 = tf.compat.v1.layers.dense(inputs=rplc_input, units=64, activation=tf.nn.relu, name='rplc_dense1')
        rplc_dense2 = tf.compat.v1.layers.dense(inputs=rplc_dense1, units=64, activation=tf.nn.relu, name='rplc_dense2')
        
        rplc_logits = tf.compat.v1.layers.dense(inputs=rplc_dense2, units=self.action_dim, name="rplc_logits")
        rplc_logits_norm = tf.nn.tanh(rplc_logits, name="plc_action")

        self.random_plc_action = rplc_logits_norm

    def run_random_policy(self, state):
        """ 
        Get action by current state 
        """
        return self.sess.run(self.random_plc_action, feed_dict = {
            self.state: state
        })

    def policy(self):
        """ 
        Create the policy model 
        """
        plc_losses=[]
        plc_train_steps=[]
        plc_input = self.state
       
        plc_dense1 = tf.compat.v1.layers.dense(inputs=plc_input, units=64, activation=tf.nn.relu, name='plc_dense1')
        plc_dense2 = tf.compat.v1.layers.dense(inputs=plc_dense1, units=64, activation=tf.nn.relu, name='plc_dense2')
        
        plc_logits = tf.compat.v1.layers.dense(inputs=plc_dense2, units=self.action_dim, name="plc_logits")
        plc_logits_norm = self.logit_normalization(self.action_mean, self.action_std, plc_logits)
        plc_logits_norm = tf.nn.sigmoid(plc_logits_norm,name="plc_action")

        self.plc_action = plc_logits_norm
        
        plc_loss = tf.keras.losses.MeanSquaredError()(self.action, self.plc_action)
        self.plc_loss = tf.math.reduce_mean(plc_loss)

        plc_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.plc_train_step = plc_optimizer.minimize(self.plc_loss)

    def run_policy(self, state):
        """ 
        Get action by current state 
        """
        return self.sess.run(self.plc_action, feed_dict = {
            self.state: state
        })


    def improve_policy(self, state, action):
        """
        Improve policy
        """
        num = len(state)
        idxs = get_shuffle_idx(num, 64)
        for idx in idxs:
            batch_s = [state[i] for i in idx]
            batch_a = [action[i] for i in idx]
            self.sess.run(self.plc_train_step, feed_dict={
            self.state: batch_s,
            self.action: batch_a
            })

    def get_policy_loss(self, state, action):
        """ 
        Get loss from policy 
        """
        return self.sess.run(self.plc_loss, feed_dict = {
            self.state: state, 
            self.action: action
        })


    def inverse_dynamic_model(self):
        """ 
        Create inverse dynamic model 
        """

        idm_logits_norm=[]
        idm_input = tf.concat([self.state, self.nstate], 1)
        
        idm_dense1 = tf.compat.v1.layers.dense(inputs=idm_input, units=64, activation=tf.nn.relu, name='idm_dense1')
        idm_dense2 = tf.compat.v1.layers.dense(inputs=idm_dense1, units=64, activation=tf.nn.relu, name='idm_dense2')
        
        idm_logits = tf.compat.v1.layers.dense(inputs=idm_dense2, units=self.action_dim, name="idm_logits")
        idm_logits_norm = self.logit_normalization(self.action_mean, self.action_std, idm_logits)
        idm_logits_norm = tf.nn.sigmoid(idm_logits_norm,name="idm_action")

        self.idm_action = idm_logits_norm
        
        idm_loss = tf.keras.losses.MeanSquaredError()(self.action, self.idm_action)
        self.idm_loss = tf.math.reduce_mean(idm_loss)
        
        idm_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.idm_train_step = idm_optimizer.minimize(self.idm_loss)

    def run_idm(self, state, nstate):
        """ 
        Get action by inverse dynamic model from current and next state 
        """
        return self.sess.run(self.idm_action, feed_dict = {
            self.state: state,
            self.nstate: nstate
        })

    def improve_idm(self, state, nstate, action):
        """
        Improve inverse dynamic model
        """
        num = len(state)
        idxs = get_shuffle_idx(num, 64)
        for idx in idxs:
            batch_s = [state[i] for i in idx]
            batch_a = [action[i] for i in idx]
            batch_ns = [nstate[i] for i in idx]
            self.sess.run(self.idm_train_step, feed_dict={
                self.state: batch_s,
                self.nstate: batch_ns,
                self.action: batch_a
            })
    def get_idm_loss(self, state, nstate, action):
        """ 
        Get loss from inverse dynamic model 
        """
        return self.sess.run(self.idm_loss, feed_dict = {
            self.state: state,
            self.nstate: nstate, 
            self.action: action
        })
    
    def state_normalization(self, x_min_arr, x_max_arr, state):
        """
        Normalize values between x_min_arr[i] and x_max_arr[i] for each feature
        """
        normalized = []
        for i in range(len(x_min_arr)):
            x_mean = x_min_arr[i]
            x_std = x_max_arr[i]
            feature = state[i]
            feature_normalized = (feature - x_mean) / x_std 
            normalized.append(feature_normalized)
        return normalized

    def state_min_max(self, states):
        """
        Find the min and max for each feature across the generated states
        """
        mins_features= [float(32)]*32
        maxs_features= [float(32)]*32

        for state in states:
            for i, val in enumerate(state):
                if(val < mins_features[i]):
                    mins_features[i] = val
                if(val > maxs_features[i]):
                    maxs_features[i] = val
        
        return mins_features, maxs_features

    def action_min_max(self, actions):
        """
        Find the min and max for each feature across the generated states
        """
        mins_features= [float(32)]*7
        maxs_features= [float(32)]*7

        for action in actions:
            for i, val in enumerate(action):
                if(val < mins_features[i]):
                    mins_features[i] = val
                if(val > maxs_features[i]):
                    maxs_features[i] = val
        
        return mins_features, maxs_features         

    def action_normalization(self, a_min, a_max, action):
        """
        Normalize actions with z-score
        """
        normalized = []
        for i in range(len(a_min)):
            x_mean = a_min[i]
            x_std = a_max[i]
            feature = action[i]
            feature_normalized = (feature - x_mean) / x_std 
            normalized.append(feature_normalized)
        return normalized

    def pre_demonstration(self):

        done = True
        States = []
        Nstates = []
        Actions = []
        norm_Actions = []
        norm_States = []
        norm_Nstates = []
        idx= 0

        for i in range(int(round(self.M / self.alpha))):
            if done:
                self.env.reset()
                s0_list = [i for i in range(200)]
                idx = np.random.choice(s0_list) * 25
                state0 = self.states[idx]
                self.env.sim.set_state_from_flattened(state0)
                self.env.sim.forward() 
                state = self.state_normalization(self.state_mean, self.state_std, state0)   

            prev_state = state
            state = np.reshape(state, [-1,self.state_dim])

            #run the untrained policy as random policy
            A = np.reshape(self.run_random_policy(state), [-1])
            A_norm = self.action_normalization(self.action_mean, self.action_std, A)
            #A = np.append(A,-1.0)

            _, _, done, _ = self.env.step(A)
            gen_state = self.env.sim.get_state().flatten()
            state = self.state_normalization(self.state_mean, self.state_std, gen_state)
            #A = np.delete(A, 6)
            
            if(state[0] > 1.15):
                done = True
                idx += 25
            
            norm_States.append(prev_state)
            norm_Nstates.append(state)
            norm_Actions.append(A_norm)

            if(idx == 5000):
                break
       
        return norm_States, norm_Nstates, norm_Actions

    def post_demonstration(self):
        """ 
        Generate the pair (s_t, s_t+1) and the associated action by policy 
        """

        done = True
        States = []
        Nstates = []
        norm_Actions = []
        norm_States = []
        norm_Nstates = []
        idx= 0

        for i in range(int(round(self.M))):
            if done:
                self.env.reset()
                s0_list = [i for i in range(200)]
                idx = np.random.choice(s0_list) * 25
                state0 = self.states[idx]
                self.env.sim.set_state_from_flattened(state0)
                self.env.sim.forward()
                state = self.state_normalization(self.state_mean, self.state_std, state0)
                
            prev_state = state
            state = np.reshape(state, [-1,self.state_dim])
            
            
            #Policy's action unormalization with expert's mins and maxs
            A = np.reshape(self.run_policy(state), [-1])
            A_unorm = self.action_unormalization(self.action_mean, self.action_std, A)
            #A_unorm = np.append(A_unorm, -1.0)
    
            _ , _, done, _ = self.env.step(A_unorm)
            gen_state = self.env.sim.get_state().flatten()
            state = self.state_normalization(self.state_mean, self.state_std, gen_state)

            if(state[0] > 1.15):
                done = True
                idx += 25
            
            norm_States.append(prev_state)
            norm_Nstates.append(state)
            norm_Actions.append(A)

            if(idx == 5000):
                break

        return norm_States, norm_Nstates, norm_Actions

    def reward(self):
        """ 
        Get the reward 
        """

        done = False
        score= 0
        count= 0
        env = suite.make(
            env_name = "Lift",
            robots = "Panda",
            controller_configs = controller_config,
            has_renderer=True,
            has_offscreen_renderer=False,
            ignore_done=True,
            use_camera_obs=False,
            reward_shaping=True,
            control_freq=20,
        )

        s0_list = [i for i in range(61)]
        idx = np.random.choice(s0_list) * 25
        env.reset()
        state = self.states[idx]
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        state = self.state_normalization(self.state_mean, self.state_std, state)

        while not done:
            state = np.reshape(state, [-1, self.state_dim])

            A = np.reshape(self.run_policy(state), [-1])
            A_unorm = self.action_unormalization(self.action_mean, self.action_std, A)
            A_unorm[2] = A_unorm[2] * 2
            #A_unorm = np.append(A_unorm,-1.0)

            
            _, reward, _, _ = env.step(A_unorm)
            env.render()
            count += 1
            state = env.sim.get_state().flatten()
            arm_point = np.array((state[1], state[2], state[3]))
            obj_point = np.array((state[10], state[11], state[12]))
            distance = np.linalg.norm(arm_point - obj_point)
            state = self.state_normalization(self.state_mean, self.state_std, state)
            
            score += reward 
            if(state[0] > 1.20):
                done = True
            
        env.close()
        return score, distance 
    
    def train(self):
        """
        Train BCO algorithm
        """

        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver = tf.compat.v1.train.Saver()

        self.plot_reward = []
        self.plot_idm_loss = []
        self.plot_policy_loss = []

        print("\nTraining...\n")
        #Set the agent's prior experiance 
        e_S, e_nS, e_A = self.pre_demonstration()
        self.improve_idm( e_S, e_nS, e_A)
        start = time.time()
        for episode in range(self.max_episodes):

            #improve policy by demonstrator
            ip_S, ip_nS = self.T_demo()
            ip_A = self.run_idm(ip_S, ip_nS)
            print(ip_A)
            self.improve_policy(ip_S, ip_A)
            policy_loss = self.get_policy_loss(ip_S, ip_A)
            
            #imitiation
            i_S, i_nS, i_A = self.post_demonstration()
            self.improve_idm(i_S, i_nS, i_A)
            idm_loss = self.get_idm_loss(i_S, i_nS, i_A)
            
            now = time.time()
            cur_score = self.reward()
            
            print('Episode: {}, reward: {}, policy loss: {}, idm loss: {}, time_sec/episode {}'.format((episode+1), cur_score, policy_loss, idm_loss, (now-start)/args.print_freq))
           
            summary_reward = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="reward", simple_value=cur_score)])
            summary_policy_loss = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="policy_loss", simple_value=policy_loss)])
            summary_idm_loss = tf.compat.v1.Summary(value=[tf.compat.v1.Summary.Value(tag="idm_loss", simple_value=idm_loss)])
            
            self.reward_writer.add_summary(summary_reward, global_step=episode)
            self.plc_loss_writer.add_summary(summary_policy_loss, global_step=episode)
            self.idm_loss_writer.add_summary(summary_idm_loss, global_step=episode)
            

            if(episode % 50 == 0):
                self.learning_rate = self.learning_rate - 0.00005

            self.plot_reward.append(cur_score)
            self.plot_idm_loss.append(idm_loss)
            self.plot_policy_loss.append(policy_loss)
        
        saver.save(self.sess, args.trained_model_dir)
        print("\nTraining just ended\n")

    def test(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, args.trained_model_dir)
        mean_score =0
        success_rate =0

        for episode in range(10):
            self.num_states, self.num_nstates, self.num_actions, self.states, self.nstates, self.actions = self.load_demonstrations()
            score, dist = self.reward()
            mean_score += score*100
            if(dist <= 1.30):
                print('\nEpisode {} with score: {}\n'.format(episode +1, score*100))
                print("Successful transition\n")
                success_rate +=1
            else:
                print('\nEpisode {} with score: {}\n'.format(episode +1, score*100))
                print("Unsuccessful transition\n")
                success_rate +=0
            
        print('Mean score= {} with success= {}\n'.format(mean_score/10, success_rate))

    def run(self):
        if args.mode == "train":
            self.num_states, self.num_nstates, self.num_actions, self.states, self.nstates, self.actions = self.load_demonstrations()
            self.norm_S, self.norm_nS = self.D_demo(self.states, self.nstates)
            self.M = self.num_states * self.alpha
            self.num_sample = self.M
            self.train()

        if args.mode == "test":
            self.test()

def main():
    bco = BCO(32, 7)
    bco.run()
    
if __name__ == "__main__":
    main()
    print("goodbye")