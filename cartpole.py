# -*- coding: utf-8 -*- 
#!/usr/bin/python2.7
import numpy as np
import pickle as pickle
#import tensorflow as tf
#import matplotlib.pyplot as plt
import math
import gym
import pandas
import json
import sys
import random

env = gym.make('CartPole-v0')
actions = range(env.action_space.n) # retorna quantidade de ações
number_episodes = 500
reward_sum = 0
number_steps = 200
number_training = 1

#Definicao do dicionario
qtable = {}
epsilon = 0.2
gamma = 0.1 # gama : é o fator de desconto, também definido entre 0 e 1. 
            # Isso modela o fato de que recompensas futuras valem menos do que recompensas imediatas.

alpha = 0.9 # alpha : é a taxa de aprendizado, definida geralmente entre 0 e 1. 
            # Configurá-la como 0 significa que os valores Q nunca são atualizados, portanto, nada é aprendido. 
            # Definir alfa para um valor alto como 0,9 significa que o aprendizado pode ocorrer rapidamente.

#Definição do Domínio das variáveis
interval_car = 2.4
interval_vel_car = 6
interval_angle = 12.4
interval_vel_angle = 6 

domain_car = 2
domain_vel_car = 4
domain_angle = 10
domain_vel_angle = 1

last_time_steps = np.ndarray(0)

car_positions = pandas.cut([-interval_car, interval_car], bins = domain_car, retbins=True)[1][1:]
vels_car = pandas.cut([-interval_vel_car, interval_vel_car], bins = domain_vel_car, retbins=True)[1][1:]
angle_positions = pandas.cut([-interval_angle, interval_angle], bins = domain_angle, retbins=True)[1][1:]
vels_angle = pandas.cut([-interval_vel_angle, interval_vel_angle], bins = domain_vel_angle, retbins=True)[1][1:]

# print 'car_positions', car_positions
# print 'vels_car', vels_car
# print 'angle_positions', angle_positions
# print 'vels_angle', vels_angle




# só cria o dicionario padrao
def create_qtable():
    qtable = {}
    for car in car_positions:
        for vel_car in vels_car:
            for angle in angle_positions:
                for vel_angle in vels_angle:                    
                    car = np.around(car, 3)
                    vel_car = np.around(vel_car, 3) # discretiza(vel_car, vels_car)
                    angle =  np.around(angle, 3) # discretiza(angle, angle_positions)
                    vel_angle = np.around(vel_angle, 3) #discretiza(vel_angle, vels_angle)
                    state = '{' + str(car) + ',' + str(vel_car) + ',' + str(angle) + ',' + str(vel_angle) + '}'
                    qtable[state] = (1 ,1)
    return qtable

# salva o dicionario (qtable) no .txt
def save_qtable(qtable, name_file):
   with open(name_file, 'w') as f:
        json.dump(qtable, f) 

# carrega o dicionario (qtable) do .txt
def load_qtable(name_file):   
    try:
        with open(name_file, 'r') as f:
            return json.load(f)
    except:
        qtable = create_qtable()
        save_qtable(qtable, name_file)
        with open(name_file) as f:
           return json.load(f)

# discretiza valor continuo para algum dos intervalos da variavel retorna posicao do intervalo em que o valor se encontra
# valor_observation é o valor da variavel (car, vel_car, angle, vel_angle) que veio do observation
# intervalo_discretizado é o intervalo discreto definido por nos para cada variavel
def discretiza(valor_observation, intervalo_discretizado):
    return np.digitize(valor_observation, intervalo_discretizado)

# transforma a "contatenacao" dos valores do observation em um estado valido no dicionario (qtable)
def construi_estado(observation):
    car = np.around(observation[0], 3)
    vel_car = np.around(observation[1], 3)
    angle = np.around(observation[2], 3)
    vel_angle = np.around(observation[3], 3)

    #print 'around{', car, vel_car, angle, vel_angle, '}'
    
    car = discretiza(car, car_positions)
    vel_car = discretiza(vel_car, vels_car)
    angle = discretiza(angle, angle_positions)
    vel_angle = discretiza(vel_angle, vels_angle)
    
    #print 'discretize{', car, vel_car, angle, vel_angle, '}'

    car = car_positions[car] 
    vel_car = vels_car[vel_car]
    angle = angle_positions[angle]
    vel_angle = vels_angle[vel_angle]

    #print 'value{', car, vel_car, angle, vel_angle, '}'

    state = '{' + str(np.around(car,3)) + ',' + str(np.around(vel_car, 3)) + ',' + str(np.around(angle, 3)) + ',' + str(np.around(vel_angle, 3)) + '}' 
    return state

# escolha uma acao dado o estado
def escolhe_action(state):
    q = qtable[state]
    maxQ = max(q)

    #tenta probabilisticamente escolher um estado aleatorio
    if np.random.random_sample() < epsilon:
        minQ = min(q)
        mag = max(abs(minQ), abs(maxQ))
        # add random values to all the actions, recalculate maxQ
        q = [q[i] + np.random.random_sample() * mag - .5 * mag for i in range(len(actions))] 
        maxQ = max(q)


    if q.count(maxQ) > 1:
        best = [i for i in range(len(actions)) if q[i] == maxQ]
        i = np.random.choice(best)
    else:
        i = q.index(maxQ)
    action = actions[i]
    return action, maxQ



# atualiza a qtable
def learning(current_state, next_state, action, reward):

    qtable[current_state][action] = np.around(qtable[current_state][action] + alpha *(reward + gamma*(max(qtable[next_state]) - qtable[current_state][action])), 5)
    #print 'Qold[', qtable[current_state] , '][' , action , ']=' , oldvalue , '\n'
    #print 'Qnew[' , current_state , '][' , action , ']=' , qtable[current_state][action] , '\n'


#printa a qtable diferente
def print_qtable(qtable):
    for key in qtable:
        print 'qtable[' + key + ']', qtable[key] , '\n'

#main
if __name__ == "__main__":

    try:
        name_file = sys.argv[1]
    
    except:
        print 'Passe como parametro o nome do arquivo que salva a Qtable'
        exit(0)

    qtable = {}
    qtable = load_qtable(name_file)

    #print_qtable(qtable)

    for training in range(number_training):
        print 'training', training
        env.reset()
        for episode in range(number_episodes):
            #print 'episode', episode
            observation = env.reset()

            for step in range(number_steps):
                env.render()
                #print 'step', step


                #discretizar observation de modo a criar um estado valido
                current_state = construi_estado(observation)

                #escolhe a melhor acao para o estado atual
                action, value_action  = escolhe_action(current_state)
                observation, reward, done, info = env.step(action)
                next_state = construi_estado(observation)
                
                #executa esta acao 
                # ????para saber qual é a melhor acao para o estaod atual?????
                #depois disso, pega o novo observation da acao tomada acima para saber qual é o proximo estado
                #aqui esta em S
                #observation, reward, done, info = env.step(action)
                # aqui esta em S'

                #print_qtable(qtable)


                if not (done):
                    # chama para aprender novamente, dado o estado atual e o proximo estado
                    # atualiza estado atual = proximo estado
                    learning(current_state, next_state, action, reward)
                    state = next_state
                else:
                    last_time_steps = np.append(last_time_steps, [int(step + 1)])
                    #print("Episode finished after {} timesteps".format(step+1))
                    break
        l = last_time_steps.tolist()
        l.sort()
        print("Overall score: {:0.2f}".format(last_time_steps.mean()))
        print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))
        print("Best score: {:0.2f}".format(max(l)))
        print("Desvio padrao: {:0.2f}".format(np.std(l)))
    save_qtable(qtable,name_file)

    #print 'after \n\n'
    #print_qtable(qtable)
       

