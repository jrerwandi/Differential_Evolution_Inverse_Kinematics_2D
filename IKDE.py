import numpy as np
import matplotlib.pyplot as plt 
import math
from pid import *


# Robot Link Length Parameter
link = [20, 30, 40, 40]
# Robot Initial Joint Values (degree)
angle = [0, 0, 0, 0]
# Target End of Effector Position
target = [0, 0, 0] 
# Create figure to plot
fig = plt.figure(1) 
ax = fig.add_subplot(1,1,1)
#ax2 = fig2.add_subplot(1,1,1)

ax.set_xlim(-150, 150)
ax.set_ylim(-150, 150)

fig2, ax2 = plt.subplots(2, 2)

# Draw Axis
def draw_axis(ax, scale=1.0, A=np.eye(4), style='-', draw_2d = False):
    xaxis = np.array([[0, 0, 0, 1], 
                      [scale, 0, 0, 1]]).T
    yaxis = np.array([[0, 0, 0, 1], 
                      [0, scale, 0, 1]]).T
    zaxis = np.array([[0, 0, 0, 1], 
                      [0, 0, scale, 1]]).T
    
    xc = A.dot( xaxis )
    yc = A.dot( yaxis )
    zc = A.dot( zaxis )
    
    if draw_2d:
        ax.plot(xc[0,:], xc[1,:], 'r' + style)
        ax.plot(yc[0,:], yc[1,:], 'g' + style)
    else:
        ax.plot(xc[0,:], xc[1,:], xc[2,:], 'r' + style)
        ax.plot(yc[0,:], yc[1,:], yc[2,:], 'g' + style)
        ax.plot(zc[0,:], zc[1,:], zc[2,:], 'b' + style)
        
        
def rotateZ(theta):
    rz = np.array([[math.cos(theta), - math.sin(theta), 0, 0],
                   [math.sin(theta), math.cos(theta), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return rz

def translate(dx, dy, dz):
    t = np.array([[1, 0, 0, dx],
                  [0, 1, 0, dy],
                  [0, 0, 1, dz],
                  [0, 0, 0, 1]])
    return t

def FK(angle, link):
    n_links = len(link)
    P = []
    P.append(np.eye(4))
    for i in range(0, n_links):
        R = rotateZ(angle[i])
        T = translate(link[i], 0, 0)
        P.append(P[-1].dot(R).dot(T))
    return P
    
def objective_function(target, thetas, link):
    P = FK(thetas, link)
    end_to_target = target - P[-1][:3, 3]
    fitness = math.sqrt(end_to_target[0] ** 2 + end_to_target[1] ** 2)
    #plt.scatter(P[-1][0,3], P[-1][1,3], marker='^')
    
    return fitness, thetas

# Cr = crossover rate
# F = mutation rate
# NP = n population
def DE(target, angle, link, n_params,Cr=0.5, F=0.5, NP=10, max_gen=300):
    
    target_vectors = np.random.rand(NP, n_params)
    target_vectors = np.interp(target_vectors, (-np.pi, np.pi), (-np.pi, np.pi))
  
    donor_vector = np.zeros(n_params)
    trial_vector = np.zeros(n_params)
    
    best_fitness = np.inf
    list_best_fitness = []
    for gen in range(max_gen):
        print("Generation :", gen)
        for pop in range(NP):
            index_choice = [i for i in range(NP) if i != pop]
            a, b, c = np.random.choice(index_choice, 3)
         
                
            donor_vector = target_vectors[a] + F * (target_vectors[b]-target_vectors[c])

            cross_points = np.random.rand(n_params) < Cr
            trial_vector = np.where(cross_points, donor_vector, target_vectors[pop])
            
            target_fitness, d = objective_function(target,target_vectors[pop],link)
            trial_fitness, e = objective_function(target,trial_vector,link)
            
            if trial_fitness < target_fitness:
                target_vectors[pop] = trial_vector.copy()
                best_fitness = trial_fitness
                angle = d
            else:
                best_fitness = target_fitness
                angle = e
         
        print("Best fitness :", best_fitness)
        '''
        P = FK(angle, link)
        for i in range(len(link)):
            start_point = P[i]
            end_point = P[i+1]
            ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            plt.scatter(target[0], target[1], marker='x', color = 'black')

        plt.pause(0.01)#plt.ion()
        plt.cla()
        '''
       
    return best_fitness, angle
def onclick(event):
    fig2.suptitle("PID", fontsize=12)
    global target, link, angle, ax
    target[0] = event.xdata
    target[1] = event.ydata
    print("Target Position : ", target)
   # plt.cla()
    
    limits = 4
    # Inverse Kinematics
    err, angle = DE(target, angle, link, limits, max_gen= 200)
    
    
    ####pid
    
    kp = 0.4
    ki = 0.8
    kd = 0.05
    x = []
    y1 = [0]
    y2 = [0]
    y3 = [0]
    y4 = [0]
    y5 = [0]
    y6 = [0]
    y7 = [0]
    y8 = [0]
    y1.pop()
    y2.pop()
    y3.pop()
    y4.pop()
    y5.pop()
    y6.pop()
    y7.pop()
    y8.pop()
    
    
    pid1 = PID(kp, ki, kd)  # default sample time : 10ms
    pid2 = PID(kp, ki, kd)  # default sample time : 10ms
    pid3 = PID(kp, ki, kd)  # default sample time : 10ms
    pid4 = PID(kp, ki, kd)  # default sample time : 10ms

    for point_num in range(30): #baru 1 joint
        t = point_num * pid1.sample_time
        set_line = angle[0]
        set_line2 = angle[1]
        set_line3 = angle[2]
        set_line4 = angle[3]
        
        output_line = pid1.update(set_line)
        output_line2 = pid2.update(set_line2)
        output_line3 = pid3.update(set_line3)
        output_line4 = pid4.update(set_line4)

        x.append(t)
        y1.append(set_line)
        y2.append(output_line)

        y3.append(set_line2)
        y4.append(output_line2)

        y5.append(set_line3)
        y6.append(output_line3)

        y7.append(set_line4)
        y8.append(output_line4)

#        print("a",output_line)
        ax2[0, 0].plot(x, y1, 'b--', x, y2, 'r')
        ax2[0, 0].set_title('joint 1')
        
        ax2[0, 1].plot(x, y3, 'b--', x, y4, 'r')
        ax2[0, 1].set_title('joint 2')
        
        ax2[1, 0].plot(x, y5, 'b--', x, y6, 'r')
        ax2[1, 0].set_title('joint 3')
        
        ax2[1, 1].plot(x, y7, 'b--', x, y8, 'r')
        ax2[1, 1].set_title('joint 4')
        
        angle2 = [output_line,output_line2,output_line3,output_line4]
        P = FK(angle2, link)
        for i in range(len(link)):
            start_point = P[i]
            end_point = P[i+1]
            ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
            ax.set_xlim(-150, 150)
            ax.set_ylim(-150, 150)
            ax.scatter(target[0], target[1], marker='x', color = 'black')


        for axs in ax2.flat:
            axs.set(xlabel='x-label', ylabel='y-label')
            
        plt.pause(0.01)#plt.ion()
        plt.cla()
        
    P = FK(angle, link)

    for i in range(len(link)):
        start_point = P[i]
        end_point = P[i+1]
        ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
        ax.set_xlim(-150, 150)
        ax.set_ylim(-150, 150)
        ax.scatter(target[0], target[1], marker='x', color = 'black')
    
    
    if (err > 1): 
       print("IK Error")
    else:
       print("IK Solved")
       
#    print("Angle :", angle) 
#    print("Target :", target)
#    print("End Effector :", P[-1][:3, 3])
#    print("Error :", err)
    fig.show()
    
    fig2.show()

def main():
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.suptitle("Differential Evolution - Inverse Kinematics", fontsize=12)
    fig2.suptitle("PID", fontsize=12)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)
    # Forward Kinematics
    P = FK(angle, link)
    # Plot Link
    for i in range(len(link)):
        start_point = P[i]
        end_point = P[i+1]
        ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
    
    plt.show()

if __name__ == "__main__":
    main()
