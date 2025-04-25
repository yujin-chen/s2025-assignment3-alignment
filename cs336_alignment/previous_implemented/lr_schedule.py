import math

def cosine_learning_rate_schedule(t, alpha_max, alpha_min, T_w, T_c):
    #warmp up
    if t < T_w:
        return (t / T_w) * alpha_max
    #cosine annealing
    elif T_w <= t <= T_c:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (alpha_max - alpha_min)
    #set constant minimum learning rate
    else:
        return alpha_min
