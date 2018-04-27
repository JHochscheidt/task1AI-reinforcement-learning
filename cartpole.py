# -*- coding:utf-8 -*-
#!/usr/local/bin/python

import numpy as np
import pickle as pickle
#import tensorflow as tf
import matplotlib.pyplot as plt
import math
import gym

env = gym.make('CartPole-v0')
#env.reset()

random_episodes = 0
reward_sum = 0

""" OBSERVATION """
POS_CART = 2.4
VEL_CART = 2.5
ANG_PEND = 2  # valor em radianos
VEL_PEND = 2.5

TAM_INTERVALO_OBJS = 10
TAM_INTERVALO_VEL = 10

# shape de 0 contem o array observation q possui todas as variaveis do problema
NUM_VAR = env.observation_space.shape[0]


qtable = { }


# print(box)
""" ACTIONS """
actions = range(env.action_space.n)
""" FIM ACTIONS """


""" REWARD """
# Recompensa é 1 para cada passo dado, incluindo o passo de terminação
# """ FIM REWARD """
REWARD = 1

""" ESTADO INICIAL """

# Todas as observations sao atribuidas a um valor aleatorio uniforme entre +- 0.05

""" FIM ESTADO INICIAL """

""" TERMINO DE UM EPISODIO """

# Episodio termina
#  se --> angulo do pendulo é maior que +-12º #um exemplo citou que o angulo seria 15º?????
# 	--> posicao do carrinho é maior que +- 2.4 ( o centro do carrinho atinge a borda da janela de exibicao
# 	--> duracao do episodio for maior que 200

""" FIM TERMINO DE UM EPISODIO """

""" RESOLUCAO DO PROBLEMA """

# Problema é resolvido quando a recompensa média é maior ou igual a 195.0 em 100 tentativas consecutivas

""" FIM RESOLUCAO DO PROBLEMA """

# state = np.zeros(4)
# for i in range(4):
#         state[i] = np.digitize(observation[i],box[i])
# print(state)


# inicia Qtable com valores todos zerados.
# funcao é chamada apenas na primeira vez que rodar o problema
def init_qtable():
    try:
        f_qtable = open('qtable.txt', 'r') #print 'tentou abrir arq'
        return f_qtable
    except:
        f_qtable = open('qtable.txt', 'w') #print 'arq nao existe. vai cria-lo'
        for i in range(10**4):
            f_qtable.writelines('0.0,0.0,0.0,0.0,0.0,0.0' + '\n')
        return f_qtable


# salva Qtable atual em um arquivo
# para posteriormente continuar o treinamento "de onde parou"
def save_qtable():
    arq_qtable = open('qtable.txt', 'w')
    close(arq_qtable)


# le QTABLE de arquivo e retorna matriz OBSERVATION e matriz ACTION
def read_qtable():
    f_qtable = open('qtable.txt', 'rb')
    qtable = np.zeros((1,6))
    
    for linha in f_qtable.readlines():
        if linha.
        linha = linha.replace('\n','') # tira \n do final
        sp_linha = np.array([linha.split(',')], dtype=float) # splita linha 
        qtable = np.insert(qtable, qtable.shape[0], sp_linha, axis=0) #insere linha do arquivo na tabela
        print(len(qtable))

    #print(qtable.shape)

    
   
f_qtable = init_qtable()


read_qtable()




#if __name__ == "__main__":
    
    
   # arq_qtable = init_qtable() # inicia a tabela em um arquivo


    #observation, action = read_qtable() #le a tabela zerada do arquivo


"""
#
# Aqui ponha o seu codigo para ler os valores dos estados, se houver
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
        print type(observation)
        if done:
#
# Aqui ponha o seu codigo para atualizar os valores dos estados apos uma execucao
#
            print("Episode finished after {} timesteps".format(t+1))
            break
#
#   Aqui ponha o seu codigo para salvar os valores dos estados calculados

"""