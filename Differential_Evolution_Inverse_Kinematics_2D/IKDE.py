import numpy as np
import matplotlib.pyplot as plt 
import math

# Robot Link Length Parameter
link = [20, 30, 40, 40]
# Robot Initial Joint Values (degree)
angle = [0, 0, 0, 0]
# Target End of Effector Position
target = [0, 0, 0] 
# Create figure to plot
fig = plt.figure() 
ax = fig.add_subplot(1,1,1)


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
        
    
       
    return best_fitness, angle
def onclick(event):
    global target, link, angle, ax
    target[0] = event.xdata
    target[1] = event.ydata
    print("Target Position : ", target)
    plt.cla()
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)

    limits = 4
    # Inverse Kinematics
    err, angle = DE(target, angle, link, limits, max_gen= 100)
    
    P = FK(angle, link)
    for i in range(len(link)):
        start_point = P[i]
        end_point = P[i+1]
        ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
        draw_axis(ax, scale=10, A=P[i+1], draw_2d=True)
    
    if (err > 1): 
       print("IK Error")
    else:
       print("IK Solved")
       
    print("Angle :", angle) 
    print("Target :", target)
    print("End Effector :", P[-1][:3, 3])
    print("Error :", err)
    plt.show()

def main():
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.suptitle("Differential Evolution - Inverse Kinematics", fontsize=12)
    ax.set_xlim(-150, 150)
    ax.set_ylim(-150, 150)

    # Forward Kinematics
    P = FK(angle, link)
    # Plot Link
    for i in range(len(link)):
        start_point = P[i]
        #print(start_point)
        end_point = P[i+1]
        ax.plot([start_point[0,3], end_point[0,3]], [start_point[1,3], end_point[1,3]], linewidth=5)
        # draw_axis(ax, scale=5, A=P[i+1], draw_2d=True)
    
    plt.show()

if __name__ == "__main__":
    main()
