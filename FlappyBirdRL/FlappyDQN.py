

import numpy as np
import random
import flappy_bird_gym
from collections import deque
from keras.layers import Input,Dense
from keras.models import load_model,save_model,Sequential
from keras.optimizers import RMSprop


#Neural Network for agent
def NeuralNetwork(input_shape,output_shape):
    model = Sequential()
    model.add(Dense(512,input_shape=input_shape,activation='relu',kernel_initializer='he_uniform'))

    model.add(Dense(256,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(64,activation='relu',kernel_initializer='he_uniform'))
    model.add(Dense(output_shape,activation='linear',kernel_initializer='he_uniform'))

    model.compile(loss='mse',optimizer = RMSprop(lr=0.0001,rho=0.95,epsilon=0.01),metrics=['accuracy'])
    print(model.summary())

    return model



# Creating Brain(class/blueprint) for the agent
class DQNAgent:
    def __init__(self):
        #Environment variables
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n #possible action that we can take in our environment i.e. jump or not jump
        self.memory = deque(maxlen=2000) #it acts as database that our model will train from. We have max of 2000 data points in here

        #Hyperparameters
        self.gamma = 0.95 #priority on immediate rewards than future rewards
        self.epsilon = 1 #taking the random action
        self.epsilon_decay = 0.9999 #Decay the epsilon as we proceed through training phase
        self.epsilon_min = 0.01 # minimum probability of decaying action to go till 0.01
        self.batch_number = 64 # amount of data-point we ae going to imput in our neural networks for training


        self.train_start = 1000
        self.jump_prob = 0.01
        self.model = NeuralNetwork(input_shape=(self.state_space,),output_shape=self.action_space)


    #Creating Acting function
    #THis act function is actually going to send action to the environment

    def act(self,state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state)) #Take the maximum value of the action performed in a given state
        return 1 if np.random.random() < self.jump_prob else 0


# Check - https://www.researchgate.net/figure/algorithm-of-deep-Q-learning_fig3_344238597 for refernce
    # Creating Learn Function
    def learn(self):
        #MAke sure we have enough data
        if len(self.memory) < self.train_start: #train_start = 1000
            return #we want to return ,we don't want to learn anythinng yet because we want more data inside of our memory

        #we want to take batch from our memory
        minibatch = random.sample(self.memory,min(len(self.memory),self.batch_number))

        #Variables to store minibatch info
        state = np.zeros((self.batch_number,self.state_space))
        next_state = np.zeros((self.batch_number,self.state_space))

        #Creating an empty list
        action,reward,done = [],[],[]

        #iterating over the  batch
        #Storing data in variables
        for i in range(self.batch_number):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])
            #Now we have our variable that will hold our data from minibatch that we took from our memory.

        #Predicting our target(y_label)
        target = self.model.predict(state) #Basically this is our label(In reinforcement learning
        # unlike supervised learning we don't have labels, so we create one)
        target_next = self.model.predict(next_state)

        for i in range(self.batch_number):
            if done[i]:
                target[i][action[i]] = reward[i]

            else:
                target[i][action[i]] = reward[i] + self.gamma *(np.amax(target_next[i]))

        self.model.fit(state,target,batch_size=self.batch_number,verbose=0)












    #Creating train function
    def train(self):
        # n episode iterations for trainning
        for i in range(self.episodes):
            #Environment Variable for training
            state = self.env.reset()
            state = np.reshape(state,[1,self.state_space]) #state will be reshaped to 1,self.state_space
            done = False
            score = 0
            self.epsilon = self.epsilon if self.epsilon * self.epsilon_decay > self.epsilon_min else self.epsilon_min
            #we want to decay our epsilon

            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)

                # reshape next state
                next_state = np.reshape(next_state,[1,self.state_space])
                score += 1 #TO increment the score

                if done:
                    reward -= 100

                self.memory.append((state,action,reward,next_state,done))
                state = next_state

                if done:
                    print('Episode:{}\nScore:{}\nEpsilon:{:}'.format(i,score,self.epsilon))

                    #Save Model
                    if score >= 1000:
                        self.model.save_model('flappybrain.h5')
                        return

                self.learn()


    # Visualising the agent

    def perform(self):
        # self.model = load_model('flappybrain.h5')
        while 1:
            state = self.env.reset()
            state = np.reshape(state,[1,self.state_space])
            done = False
            score = 0

            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state,reward,done,info = self.env.step(action)
                state = np.reshape(next_state,[1,self.state_space])
                score += 1

                print("Current Score:{}".format(score))

                if done:
                    print('DEAD')
                    break











if __name__ == '__main__':
    agent = DQNAgent()
    # agent.train()
    agent.perform()
