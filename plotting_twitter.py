import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

# style.use('fivethirtyeight')
style.use('ggplot')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pull_data = open('C:/Users/jvw/Documents/Python Scripts/nltk/twitter-out.txt','r').read()
    lines = pull_data.split('\n')
    xs = []
    ys = []
    x = 0
    y = 0
    for l in lines[-200:]:
        x += 1
        if "pos" in l:
            y += 1
        elif "neg" in l:
            y -= 1
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()
