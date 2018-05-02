# -*- coding: utf-8 -*- 
#!/usr/bin/python2.7
import numpy as np
import matplotlib.pyplot as plt
import gym
import pandas
import json
import sys

#Definição do intervalo das variáveis
interval_car = 2.5
interval_vel_car = 6
interval_angle = 13
interval_vel_angle = 6 
#Definição do Domínio das variáveis
domain_car = 1
domain_vel_car = 10
domain_angle = 10
domain_vel_angle = 10

#Gerando valores para os índices dos intervalos
car_positions = pandas.cut([-interval_car, interval_car], bins = domain_car, retbins=True)[1][1:]
vels_car = pandas.cut([-interval_vel_car, interval_vel_car], bins = domain_vel_car, retbins=True)[1][1:]
angle_positions = pandas.cut([-interval_angle, interval_angle], bins = domain_angle, retbins=True)[1][1:]
vels_angle = pandas.cut([-interval_vel_angle, interval_vel_angle], bins = domain_vel_angle, retbins=True)[1][1:]

#casas decimais, para arredondamento
around = 3

# cria o dicionario padrao
def create_qtable():
    qtable = {}
    for car in car_positions:
        for vel_car in vels_car:
            for angle in angle_positions:
                for vel_angle in vels_angle:                    
                    car = np.around(car, around)
                    vel_car = np.around(vel_car, around) 
                    angle =  np.around(angle, around) 
                    vel_angle = np.around(vel_angle, around) 
                    state = '{' + str(car) + ',' + str(vel_car) + ',' + str(angle) + ',' + str(vel_angle) + '}'
                    qtable[state] = (1 ,1)
    return qtable

# salva o dicionario (qtable) no arquivo
def save_qtable(qtable, name_file):
   with open(name_file, 'w') as f:
        json.dump(qtable, f) 

# carrega o dicionario (qtable) do arquivo
def load_qtable(name_file):   
    try:
        with open(name_file, 'r') as f:
            return json.load(f)
    except:
        qtable = create_qtable()
        save_qtable(qtable, name_file)
        with open(name_file) as f:
           return json.load(f)

# discretiza valor continuo para algum dos intervalos da variavel
# retorna posicao do intervalo em que o valor se encaixa
# valor_observation é o valor da variavel (car, vel_car, angle, vel_angle) que veio do observation
# intervalo_discretizado é o intervalo discreto definido por nos para cada variavel
def discretize(valor_observation, intervalo_discretizado):
    return np.digitize(valor_observation, intervalo_discretizado)

# transforma a "concatenacao" dos valores do observation em um estado valido no dicionario (qtable)
def build_state(observation):
    car = discretize(observation[0], car_positions)
    vel_car = discretize(observation[1], vels_car)
    angle = discretize(observation[2], angle_positions)
    vel_angle = discretize(observation[3], vels_angle)
    
    car = car_positions[car] 
    vel_car = vels_car[vel_car]
    angle = angle_positions[angle]
    vel_angle = vels_angle[vel_angle]

    state = '{' + str(np.around(car, around)) + ',' + str(np.around(vel_car, around)) + ',' + str(np.around(angle, around)) + ',' + str(np.around(vel_angle, around)) + '}' 
    return state

# escolha uma acao dado o estado
def action_choice(state):
    q = qtable[state]
    maxQ = max(q)
    i = q.index(maxQ)
    action = actions[i]
    return action, maxQ

# atualiza a qtable
def learning(current_state, next_state, action, reward):
    qtable[current_state][action] = \
        np.around(qtable[current_state][action] + \
        alpha *(reward + gamma*(max(qtable[next_state])) - qtable[current_state][action]), around)

#verifica media de 100 valores consecutivos
def average_hundred_consecutive_score(scores):
    if len(scores) >= 100: #se vetor é maior que 100
        for i in range(len(scores)):
            if(len(scores[i:]) >= 100): #se vetor é maior que sem
                media = scores[i:i+100].mean()
                if media >= 195.0:
                    return True, i
    return False, len(scores)

#main
if __name__ == "__main__":
    try:
        name_file = sys.argv[1] 
    except:
        print 'Passe como parametro o nome do arquivo que salva a Qtable'
        exit(0)
    env = gym.make('CartPole-v0')
    actions = range(env.action_space.n) # retorna quantidade de ações

    """"""""""""""""""""""""""""""""
    """  PARAMETROS TREINAMENTO  """
    """"""""""""""""""""""""""""""""
    number_episodes = 100   #numero de episodios
    reward_sum = 0          #somatorio das recompensas
    number_steps = 200      #numero de passos                                           #este valor nao pode ser alterado
     
    average_reward = 195                #media das recompensas ao final de todos os episodios       #este numero nao pode ser alterado
    last_time_steps = np.ndarray(0)     #array que guarda quantos passos ocorreram em cada episodio. 
    reward_steps = np.ndarray(0)        #array que guarda a recompensa total de cada episodio
    episodes = np.ndarray(0)            #array utilizado apenas para montar os graficos

    gamma = 0.3     # fator de desconto[0,1]. Define se recompensas futuras valem  valem menos do que recompensas imediatas.
    alpha = 0.7     # taxa de aprendizado[0,1]. --> 0 significa que os valores Q nunca são atualizados, portanto, nada é aprendido. 
                    #                           --> Definir para um valor alto como 0,9 significa que o aprendizado pode ocorrer rapidamente.

    """"""""""""""""""""""""""""""""
    """FIM PARAMETROS TREINAMENTO"""
    """"""""""""""""""""""""""""""""
    #Definicao do dicionario
    qtable = {}
    #carrega qtable do arquivo ou cria a qtable, salva e depois carrega
    qtable = load_qtable(name_file)

    for episode in range(number_episodes):
        #print 'episode', episode
        episodes = np.append(episodes, episode)
        observation = env.reset()
        reward_sum = 0
        for step in range(number_steps):
            #env.render()
            #construi estado, escolhe uma ação e faz essa ação afim de verificar observation para depois pegar o proximo estado
            current_state = build_state(observation)
            action, value_action  = action_choice(current_state)
            observation, reward, done, info = env.step(action)
            next_state = build_state(observation)

            if done:
                if step != (number_steps-1):
                    learning(current_state, next_state, action, -10) #só atualiza o estado que fez com que terminasse o episodio com uma recompensa negativa
                    reward_sum += -10
                else: 
                    reward_sum+=reward 
                reward_steps = np.append(reward_steps,reward_sum)      #adiciona recompensa ao vetor de recompensas
                last_time_steps = np.append(last_time_steps, step+1)#adiciona valor do ultimo passo do episodio ao vetor
                #print("Episode finished after {} timesteps".format(step+1))
                break
            else:
                reward_sum += reward
                learning(current_state, next_state, action, reward) # chama para aprender novamente, dado o estado atual e o proximo estado
                state = next_state                                  # atualiza estado atual = proximo estado
                reward_steps = np.append(reward_steps,reward_sum)          #adiciona recompensa ao vetor de recompensas
        #fim laço step
    #fim laço episode          

    avg_rw_all_episodes = reward_steps.mean()  #media de recompensa entre todos os episodios
    avg_last_time_steps = last_time_steps.mean() #media de passos entre todos os episodios
    std_dev_last_time_steps = np.std(last_time_steps) #desvio padrao
    
    avg_all_steps = np.ndarray(number_episodes)
    avg_all_steps[:] = avg_last_time_steps

    avg_rw_all_ep = np.ndarray(number_episodes)
    avg_rw_all_ep[:] = avg_rw_all_episodes

    solved, i = average_hundred_consecutive_score(last_time_steps)
    if solved:
        print '[' + str(i) + '|' + str(i+100) + ']\n' + str(last_time_steps[i:i+100]) + '\n'

    """ 
    GRAFICOS 
    """
    fig = plt.figure() #figura em que serão desenhados os gráficos
    rect = fig.patch
    ax1 = fig.add_subplot(1,1,1) #figura terá apenas 1 gráfico
    ax1.set_title('Resultado treinamento')  #Título do gráfico
    ax1.set_xlabel('episode')               #Eixo X
    ax1.set_ylabel('step')                  #Eixo y
   
    ax1.plot(episodes, last_time_steps , color='r', label='Qt steps') #plota o ultimo passo de cada episodio
    ax1.plot(episodes, avg_all_steps, color='b', label='Avg all episodes') #plota a média entre todos os episódios
    if solved:
        avg_best_hundred_consecutive_score = last_time_steps[i:i+100].mean()
        avg_best_hd_con_score = np.ndarray(number_episodes)
        avg_best_hd_con_score[i:i+100] = avg_best_hundred_consecutive_score
        ax1.plot(episodes[i:i+100], avg_best_hd_con_score[i:i+100], color='g', label='Avg best 100 consecutives') #plota a média entre 100 melhores os episódios
    ax1.legend(loc='upper left')
    
    plt.show()
    plt.savefig("resultado_final3.png")

    print 'number_episodes', number_episodes
    print 'alpha', alpha
    print 'gamma', gamma
    print("\nAverage Overall score: {:0.2f}".format(avg_last_time_steps))
    print("Best score: {:0.2f}".format(max(last_time_steps)))
    print("Standard Deviation: {:0.2f}".format(np.std(last_time_steps)))
       
    save_qtable(qtable,name_file)
