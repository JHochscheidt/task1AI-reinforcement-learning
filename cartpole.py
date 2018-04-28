# -*- coding: utf-8 -*- 
import numpy as np
import pickle as pickle
#import tensorflow as tf
#import matplotlib.pyplot as plt
import math
import gym
import pandas
import json
env = gym.make('CartPole-v0')
env.reset()
random_episodes = 0
reward_sum = 0
#
# Aqui ponha o seu codigo para ler os valores dos estados, se houver

#Definicao do dicionario
qtable = {}

#Definição do Domínio das variáveis

interval_car = 2.4
interval_vel_car = 5
interval_angle = 42
interval_vel_angle = 5

domain_car = 10
domain_vel_car = 10
domain_angle = 10
domain_vel_angle = 10


car_positions = pandas.cut([-interval_car, interval_car], bins = domain_car, retbins=True)[1][1:-1]
vels_car = pandas.cut([-interval_vel_car, interval_vel_car], bins = domain_vel_car, retbins=True)[1][1:-1]
angle_positions = pandas.cut([-interval_angle, interval_angle], bins = domain_angle, retbins=True)[1][1:-1]
vels_angle = pandas.cut([-interval_vel_angle, interval_vel_angle], bins = domain_vel_angle, retbins=True)[1][1:-1]
# print 'car' , car_positions
# print 'vel_c' , vels_car
# print 'angle' , angle_positions
# print 'vel_ang' , vels_angle

# só cria o dicionario padrao
def create_qtable():
    qtable = {}
    for car in car_positions:
        for vel_car in vels_car:
            for angle in angle_positions:
                for vel_angle in vels_angle:                    
                    car = np.around(car, 2)
                    vel_car = np.around(vel_car, 5) # discretiza(vel_car, vels_car)
                    angle =  np.around(angle, 2) # discretiza(angle, angle_positions)
                    vel_angle = np.around(vel_angle, 5) #discretiza(vel_angle, vels_angle)

                    # gera valor rand entre 0.0 e 1.0
                    val_left = np.around (np.random.random_sample() , 5)
                    val_right = np.around (np.random.random_sample(), 5)
                    state = '{' + str(car) + ',' + str(vel_car) + ',' + str(angle) + ',' + str(vel_angle) + '}'
                    qtable[state] = (val_left, val_right)
    return qtable

# salva o dicionario (qtable) no .txt
def save_qtable(qtable):
    with open('qtable.txt', 'w') as f:
        json.dump(qtable, f)

# carrega o dicionario (qtable) do .txt
def load_qtable():
    with open('qtable.txt') as f:
        return json.load(f)


# discretiza valor continuo para algum dos intervalos da variavel retorna posicao do intervalo em que o valor se encontra
# valor_observation é o valor da variavel (car, vel_car, angle, vel_angle) que veio do observation
# intervalo_discretizado é o intervalo discreto definido por nos para cada variavel
def discretiza(valor_observation, intervalo_discretizado):
    return np.digitize(valor_observation, intervalo_discretizado)

# transforma a "contatenacao" dos valores do observation em um estado valido no dicionario (qtable)
def construi_estado(observation):
    car = discretiza(observation[0], car_positions)
    vel_car = discretiza(observation[1], vels_car)
    angle = discretiza(observation[2], angle_positions)
    vel_angle = discretiza(observation[3], vels_angle)


    car = car_positions[car] 
    vel_car = vels_car[vel_car]
    angle = angle_positions[angle]
    vel_angle = vels_angle[vel_angle]

    state = '{' + str( np.around(car, 2)) + ',' + str(np.around(vel_car, 5)) + ',' + str(np.around(angle,2)) + ',' + str(np.around(vel_angle, 5)) + '}' 
    print 'state',  state



qtable = create_qtable()

#print qtable

save_qtable(qtable)

#qtable = load_qtable()
#print qtable


observation = env.reset()

print 'obs' ,observation

construi_estado(observation)



"""
#
while random_episodes < 100:
    observation = env.reset()
    for t in range(200):
        env.render()
        #print(observation)
#
# Aqui ponho o seu codigo para salvar a observacao, estado
#
#
# Aqui ponha o seu codigo para escolher a melhor acao dependendo do estado atual
#
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
#
# Aqui ponha o seu codigo para atualizar os valores dos estados apos uma execucao
#
            print("Episode finished after {} timesteps".format(t+1))
            break

#
#   Aqui ponha o seu codigo para salvar os valores dos estados calculados
"""