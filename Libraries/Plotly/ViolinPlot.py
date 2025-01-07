import plotly.graph_objects as go
import plotly.express as px

df_tips = px.data.tips()

fig = go.Figure()

fig.add_trace(go.Violin(x=df_tips['day'][df_tips['smoker'] == 'Yes'],
                        y=df_tips['total_bill'][df_tips['smoker'] == 'Yes'],
                        legendgroup='Yes',
                        scalegroup='Yes',
                        name='Yes',
                        side='negative',
                        line_color='blue'
                        )
              )

fig.add_trace(go.Violin(x=df_tips['day'][df_tips['smoker'] == 'No'],
                        y=df_tips['total_bill'][df_tips['smoker'] == 'No'],
                        legendgroup='Yes',
                        scalegroup='Yes',
                        name='No',
                        side='positive',
                        line_color='red'
                        )
              )

fig.show()
