# recreating https://twitter.com/NateB_Panic/status/1636811443612860417/photo/1

import numpy as np
import plotly
import plotly.graph_objects as go
import numpy as np

p_values = [0.01, 0.05,0.1,0.2, 0.3, 0.5]  # set the values of p
x = np.arange(0, 30, 1)  # generate an array of x values from 0 to 10

fig = go.Figure()  # create a new plotly figure

for p in p_values:
    y = (1- ((1- p) ** x))*100  
    name = f"p = {p} (CDC estimate)" if p==0.2 else f"p = {p}"
    fig.add_trace(go.Scatter(x=x, y=y, name=name)) 

fig.update_layout(title="How long  util I get long covid for different values of p [y = (1-p)**x] ",
                  xaxis_title="Numbers of times infected",
                  yaxis_title="The risk of getting Long Covid (%)")  

# fig.show()  # show the figure
plotly.offline.init_notebook_mode(connected=True)
plotly.offline.plot(fig)
