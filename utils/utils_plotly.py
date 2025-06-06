import plotly.graph_objects as go
import numpy as np


'''
    This file is dedicated to creating plotly plots for traversal network visualization.
'''


def MNIST_OneDigit_Plotter(pca_result, center_shift,
                           manifold, network_params,
                           digit, save_fig_path,
                           R_denoising, sigma, N,
                           save_figure=True):
    
    # unpack params
    # [mean_MT_SE, mean_data_SE, all_MT_SE] = errors
    [local_params, nbrs_info, misc] = network_params
    [Q, T, S_collection, P, Xi] = local_params
    [N1, W1, N0, W0] = nbrs_info
    [tangent_colors,D, d, M, P] = misc

    # Create empty lists for traces and buttons
    traces = []
    buttons = []

    # Pre-compute all embeddings once and store them
    embeddings = {}
    for m in range(M):
        embeddings[m] = manifold.visualization_embedding(Q[m])[0]

    # 1. First add the MNIST manifold trace
    manifold_trace = go.Scatter3d(
        x=pca_result[0, :] + center_shift[0],
        y=pca_result[1, :] + center_shift[1],
        z=pca_result[2, :] + center_shift[2],
        mode='markers',
        name='MNIST Manifold',
        marker=dict(
            size=4,
            opacity=0.1,
            color='green',
        ),
        visible=True
    )
    traces.append(manifold_trace)

    # 2. Add landmarks
    points = np.array([embeddings[m] for m in range(M)])
    landmark_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        ),
        name='Landmarks',
        visible=True
    )
    traces.append(landmark_trace)

    # 3. Add first-order edges
    edge_x1, edge_y1, edge_z1 = [], [], []
    for m in range(M):
        for n in N1[m]:
            start = embeddings[m]
            end = embeddings[n]
            edge_x1.extend([start[0], end[0], None])
            edge_y1.extend([start[1], end[1], None])
            edge_z1.extend([start[2], end[2], None])

    foe_trace = go.Scatter3d(
        x=edge_x1, y=edge_y1, z=edge_z1,
        mode='lines',
        line=dict(color='blue', width=2),
        name='First-order edges',
        opacity=0.6,
        visible=True
    )
    traces.append(foe_trace)

    # 4. Add zero-order edges
    edge_x0, edge_y0, edge_z0 = [], [], []
    for m in range(M):
        for n in N0[m]:
            start = embeddings[m]
            end = embeddings[n]
            edge_x0.extend([start[0], end[0], None])
            edge_y0.extend([start[1], end[1], None])
            edge_z0.extend([start[2], end[2], None])

    zoe_trace = go.Scatter3d(
        x=edge_x0, y=edge_y0, z=edge_z0,
        mode='lines',
        line=dict(color='green', width=2),
        name='Zero-order edges',
        opacity=0.6,
        visible=True
    )
    traces.append(zoe_trace)

    # Create buttons for each component
    manifold_button = dict(
        label='Manifold',
        method='update',
        args=[{'visible': [True, False, False, False]},
            {'showlegend': True}]
    )
    buttons.append(manifold_button)

    landmark_button = dict(
        label='Landmarks',
        method='update',
        args=[{'visible': [False, True, False, False]},
            {'showlegend': True}]
    )
    buttons.append(landmark_button)

    foe_button = dict(
        label='First-order Edges',
        method='update',
        args=[{'visible': [False, False, True, False]},
            {'showlegend': True}]
    )
    buttons.append(foe_button)

    zoe_button = dict(
        label='Zero-order Edges',
        method='update',
        args=[{'visible': [False, False, False, True]},
            {'showlegend': True}]
    )
    buttons.append(zoe_button)

    # Add combined view buttons
    manifold_landmarks_button = dict(
        label='Manifold + Landmarks',
        method='update',
        args=[{'visible': [True, True, False, False]},
            {'showlegend': True}]
    )
    buttons.append(manifold_landmarks_button)

    landmarks_edges_button = dict(
        label='Landmarks + Edges',
        method='update',
        args=[{'visible': [False, True, True, True]},
            {'showlegend': True}]
    )
    buttons.append(landmarks_edges_button)

    show_all_button = dict(
        label='Show All',
        method='update',
        args=[{'visible': [True, True, True, True]},
            {'showlegend': True}]
    )
    buttons.append(show_all_button)

    # Create figure with all traces
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        # title=f'Digit {digit}, R_denoising={R_denoising}, R_1st_order_nbhd={R_1st_order_nbrs}, sigma={sigma}, # of landmarks={M}, N={N}',
        title=f'PCA visualization of MNIST Digit {digit},N={N},M={M},max(P)={max(P)}',
        scene=dict(
            xaxis_title='PCA feature 1',
            yaxis_title='PCA feature 2',
            zaxis_title='PCA feature 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            dragmode='zoom'
        ),
        updatemenus=[dict(
            type="buttons",
            direction="down",
            x=-0.1,
            y=1,
            xanchor="right",
            yanchor="top",
            pad={"r": 10, "t": 10},
            showactive=True,
            buttons=buttons
        )],
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=150, r=150)
    )

    # Update axes ranges
    max_range = max([
        np.abs(pca_result + center_shift[:, np.newaxis]).max() * 1.1
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[-max_range, max_range])
        )
    )

    if save_figure:
        # Save the figure as HTML
        fig.write_html(f"{save_fig_path}/mnist_visualization_digit{digit}R{R_denoising}_sigma{sigma}_M{M}_N{N}.html")

    # Display the figure
    fig.show()



def GW_plotter(pca_result, center_shift,
                           manifold, network_params,
                           save_fig_path,
                           R_denoising, sigma, N,
                           save_figure=True):
    
    # unpack params
    # [mean_MT_SE, mean_data_SE, all_MT_SE] = errors
    [local_params, nbrs_info, misc] = network_params
    [Q, T, S_collection, P, Xi] = local_params
    [N1, W1, N0, W0] = nbrs_info
    [tangent_colors,D, d, M, P] = misc

    # Create empty lists for traces and buttons
    traces = []
    buttons = []

    pca_result_tr = pca_result.T

    # Pre-compute all embeddings once and store them
    embeddings = {}
    for m in range(M):
        embeddings[m] = manifold.visualization_embedding(Q[m])[0]

    # 1. First add the GW manifold trace
    manifold_trace = go.Scatter3d(
        x=pca_result_tr[0, :] + center_shift[0],
        y=pca_result_tr[1, :] + center_shift[1],
        z=pca_result_tr[2, :] + center_shift[2],
        mode='markers',
        name='MNIST Manifold',
        marker=dict(
            size=4,
            opacity=0.1,
            color='orange',
        ),
        visible=True
    )
    traces.append(manifold_trace)

    # 2. Add landmarks
    points = np.array([embeddings[m] for m in range(M)])
    landmark_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        ),
        name='Landmarks',
        visible=True
    )
    traces.append(landmark_trace)

    # 3. Add first-order edges
    edge_x1, edge_y1, edge_z1 = [], [], []
    for m in range(M):
        for n in N1[m]:
            start = embeddings[m]
            end = embeddings[n]
            edge_x1.extend([start[0], end[0], None])
            edge_y1.extend([start[1], end[1], None])
            edge_z1.extend([start[2], end[2], None])

    foe_trace = go.Scatter3d(
        x=edge_x1, y=edge_y1, z=edge_z1,
        mode='lines',
        line=dict(color='blue', width=2),
        name='First-order edges',
        opacity=0.6,
        visible=True
    )
    traces.append(foe_trace)

    # 4. Add zero-order edges
    edge_x0, edge_y0, edge_z0 = [], [], []
    for m in range(M):
        for n in N0[m]:
            start = embeddings[m]
            end = embeddings[n]
            edge_x0.extend([start[0], end[0], None])
            edge_y0.extend([start[1], end[1], None])
            edge_z0.extend([start[2], end[2], None])

    zoe_trace = go.Scatter3d(
        x=edge_x0, y=edge_y0, z=edge_z0,
        mode='lines',
        line=dict(color='green', width=2),
        name='Zero-order edges',
        opacity=0.6,
        visible=True
    )
    traces.append(zoe_trace)

    # Create buttons for each component
    manifold_button = dict(
        label='Manifold',
        method='update',
        args=[{'visible': [True, False, False, False]},
            {'showlegend': True}]
    )
    buttons.append(manifold_button)

    landmark_button = dict(
        label='Landmarks',
        method='update',
        args=[{'visible': [False, True, False, False]},
            {'showlegend': True}]
    )
    buttons.append(landmark_button)

    foe_button = dict(
        label='First-order Edges',
        method='update',
        args=[{'visible': [False, False, True, False]},
            {'showlegend': True}]
    )
    buttons.append(foe_button)

    zoe_button = dict(
        label='Zero-order Edges',
        method='update',
        args=[{'visible': [False, False, False, True]},
            {'showlegend': True}]
    )
    buttons.append(zoe_button)

    # Add combined view buttons
    manifold_landmarks_button = dict(
        label='Manifold + Landmarks',
        method='update',
        args=[{'visible': [True, True, False, False]},
            {'showlegend': True}]
    )
    buttons.append(manifold_landmarks_button)

    landmarks_edges_button = dict(
        label='Landmarks + Edges',
        method='update',
        args=[{'visible': [False, True, True, True]},
            {'showlegend': True}]
    )
    buttons.append(landmarks_edges_button)

    show_all_button = dict(
        label='Show All',
        method='update',
        args=[{'visible': [True, True, True, True]},
            {'showlegend': True}]
    )
    buttons.append(show_all_button)

    # Create figure with all traces
    fig = go.Figure(data=traces)

    # Update layout
    fig.update_layout(
        title=f'R_denoising={R_denoising}, sigma={sigma}, # of landmarks={M}, N={N}',
        scene=dict(
            xaxis_title='PCA feature 1',
            yaxis_title='PCA feature 2',
            zaxis_title='PCA feature 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            dragmode='zoom'
        ),
        updatemenus=[dict(
            type="buttons",
            direction="down",
            x=-0.1,
            y=1,
            xanchor="right",
            yanchor="top",
            pad={"r": 10, "t": 10},
            showactive=True,
            buttons=buttons
        )],
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=150, r=150)
    )

    # Update axes ranges
    max_range = max([
        np.abs(pca_result_tr + center_shift[:, np.newaxis]).max() * 1.1
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[-max_range, max_range])
        )
    )

    # Save the figure as HTML
    if save_figure:
        fig.write_html(f"{save_fig_path}/mnist_visualization_R{R_denoising}_sigma{sigma}_M{M}_N{N}.html")

    # Display the figure
    fig.show()



def GW_plotter_with_i(pca_result, center_shift,
                           manifold, network_params,
                           save_fig_path,
                           R_denoising, sigma, N,
                           save_figure=True,
                           highlight_index=None):  # Added parameter to highlight a specific point
    
    # unpack params
    # [mean_MT_SE, mean_data_SE, all_MT_SE] = errors
    [local_params, nbrs_info, misc] = network_params
    [Q, T, S_collection, P, Xi] = local_params
    [N1, W1, N0, W0] = nbrs_info
    [tangent_colors,D, d, M, P] = misc

    # Create empty lists for traces and buttons
    traces = []
    buttons = []

    pca_result_tr = pca_result.T

    # Pre-compute all embeddings once and store them
    embeddings = {}
    for m in range(M):
        embeddings[m] = manifold.visualization_embedding(Q[m])[0]

    # 1. First add the GW manifold trace
    manifold_trace = go.Scatter3d(
        x=pca_result_tr[0, :] + center_shift[0],
        y=pca_result_tr[1, :] + center_shift[1],
        z=pca_result_tr[2, :] + center_shift[2],
        mode='markers',
        name='MNIST Manifold',
        marker=dict(
            size=4,
            opacity=0.1,
            color='orange',
        ),
        visible=True
    )
    traces.append(manifold_trace)

    # 2. Add landmarks
    points = np.array([embeddings[m] for m in range(M)])
    landmark_trace = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.3
        ),
        name='Landmarks',
        visible=True
    )
    traces.append(landmark_trace)

    # 3. Add first-order edges
    edge_x1, edge_y1, edge_z1 = [], [], []
    for m in range(M):
        for n in N1[m]:
            start = embeddings[m]
            end = embeddings[n]
            edge_x1.extend([start[0], end[0], None])
            edge_y1.extend([start[1], end[1], None])
            edge_z1.extend([start[2], end[2], None])

    foe_trace = go.Scatter3d(
        x=edge_x1, y=edge_y1, z=edge_z1,
        mode='lines',
        line=dict(color='blue', width=2),
        name='First-order edges',
        opacity=0.6,
        visible=True
    )
    traces.append(foe_trace)

    # 4. Add zero-order edges
    edge_x0, edge_y0, edge_z0 = [], [], []
    for m in range(M):
        for n in N0[m]:
            start = embeddings[m]
            end = embeddings[n]
            edge_x0.extend([start[0], end[0], None])
            edge_y0.extend([start[1], end[1], None])
            edge_z0.extend([start[2], end[2], None])

    zoe_trace = go.Scatter3d(
        x=edge_x0, y=edge_y0, z=edge_z0,
        mode='lines',
        line=dict(color='green', width=2),
        name='Zero-order edges',
        opacity=0.6,
        visible=True
    )
    traces.append(zoe_trace)
    
    # 5. Add highlighted point q_i if index is provided
    if highlight_index is not None and 0 <= highlight_index < M:
        q_i = embeddings[highlight_index]
        highlighted_point_trace = go.Scatter3d(
            x=[q_i[0]],
            y=[q_i[1]],
            z=[q_i[2]],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=1.0,
                symbol='circle'
            ),
            name=f'q_{highlight_index}',
            visible=True
        )
        traces.append(highlighted_point_trace)

    # Create buttons for each component, updated to include highlighted point
    visible_states = []
    
    # Manifold only
    visible_states.append([True, False, False, False] + ([True] if highlight_index is not None else []))
    
    # Landmarks only
    visible_states.append([False, True, False, False] + ([True] if highlight_index is not None else []))
    
    # First-order Edges only
    visible_states.append([False, False, True, False] + ([True] if highlight_index is not None else []))
    
    # Zero-order Edges only
    visible_states.append([False, False, False, True] + ([True] if highlight_index is not None else []))
    
    # Manifold + Landmarks
    visible_states.append([True, True, False, False] + ([True] if highlight_index is not None else []))
    
    # Landmarks + Edges
    visible_states.append([False, True, True, True] + ([True] if highlight_index is not None else []))
    
    # Show All
    visible_states.append([True, True, True, True] + ([True] if highlight_index is not None else []))

    button_labels = [
        'Manifold', 
        'Landmarks', 
        'First-order Edges', 
        'Zero-order Edges',
        'Manifold + Landmarks',
        'Landmarks + Edges',
        'Show All'
    ]
    
    buttons = []
    for i, label in enumerate(button_labels):
        buttons.append(dict(
            label=label,
            method='update',
            args=[{'visible': visible_states[i]},
                  {'showlegend': True}]
        ))

    # Create figure with all traces
    fig = go.Figure(data=traces)

    # Update layout
    title_text = f'R_denoising={R_denoising}, sigma={sigma}, # of landmarks={M}, N={N}'
    if highlight_index is not None:
        title_text += f', Highlighting q_{highlight_index}'
        
    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title='PCA feature 1',
            yaxis_title='PCA feature 2',
            zaxis_title='PCA feature 3',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            dragmode='zoom'
        ),
        updatemenus=[dict(
            type="buttons",
            direction="down",
            x=-0.1,
            y=1,
            xanchor="right",
            yanchor="top",
            pad={"r": 10, "t": 10},
            showactive=True,
            buttons=buttons
        )],
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="middle",
            y=0.5,
            xanchor="right",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        margin=dict(l=150, r=150)
    )

    # Update axes ranges
    max_range = max([
        np.abs(pca_result_tr + center_shift[:, np.newaxis]).max() * 1.1
    ])
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-max_range, max_range]),
            yaxis=dict(range=[-max_range, max_range]),
            zaxis=dict(range=[-max_range, max_range])
        )
    )

    # Save the figure as HTML
    if save_figure:
        file_suffix = f"_q{highlight_index}" if highlight_index is not None else ""
        fig.write_html(f"{save_fig_path}/mnist_visualization_R{R_denoising}_sigma{sigma}_M{M}_N{N}{file_suffix}.html")

    # Display the figure
    fig.show()
