#!/usr/bin/env python
from CDQN import CDQN
import tensorflow as tf
import numpy as np
import gym

def atari_breakout():
    input_dims = (128,128,9)
    output_dims = 2
    hidden_layer_sizes = [256,256,256]
    model = CDQN(input_dims, output_dims, hidden_layer_sizes,.99 )
    target_network = CDQN(input_dims,output_dims,hidden_layer_sizes,.99)
    episode_round = 0
    max_episode_iterations = 120
    max_episodes = 250
    done = False
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    model.set_session(sess)
    target_network.set_session(sess)
    env = gym.make('Breakout-v0')
    s0 = env.reset()
    env.step(0)
    for ep in range(max_episodes):
        env.step(1)
        s0 = skimage.transform.resize(s0, (128,128,3))
        states = np.dstack((s0,s0,s0))
        state = skimage.transform.resize(states, (128,128,9))

        while not done and episode_round <= max_episode_iterations: 
            raw_action = list(model.predict(np.reshape(states, (1,states.shape[0], states.shape[1], states.shape[2]))))
            action = int(np.argmax(raw_action)) + 2
            new_states, reward, done, _ = env.step(action)
            new_states = skimage.transform.resize(new_states, (128,128,3))
            new_states = np.dstack((new_states,states[:,:,:-3]))
            model.add_experience(states, action, reward, new_states, done)
            model.train(target_network)
            states = new_states
            episode_round += 1
            env.render()
            time.sleep(.2)
            print("Round ", episode_round,", Action: ",raw_action, action, ", Reward: ", reward, ", Num Experiences: ", len(model.experience))
        episode_round = 0
        env.close()
        s0 = env.reset()
        done = False
        print("Round is: ", ep)

def main():
    atari_breakout()

if __name__ == "__main__":
    main()


