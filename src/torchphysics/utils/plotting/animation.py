'''This file contains different functions for animating the output of 
the diffeqmodel-class
'''
import matplotlib.pyplot as plt
import matplotlib.tri as plt_tri
from matplotlib import cm, colors
from matplotlib import animation as anim
import numpy as np
import torch

from .plot import (_compute_output_shape, _create_info_text, 
                   _create_figure_and_axis, _triangulation_of_domain)
from ...problem.samplers.plot_samplers import AnimationSampler
from ..helper import prepare_user_fun_input


def animate(model, ani_function, ani_sampler, ani_speed=50, angle=[30, 30],
            ani_type=''):
    '''Main function for animations
    
    Parameters
    ----------
    model : DiffEqModel
        A neural network of which the output should be animated.
    plot_variable : Variabale.
        The main variable over which the solution should be animated. 
    domain_points : int 
        The number of points that should be used for the domain discretization.
    frame_number : int
        Number of frames, of the animation. Will be sampled equidistant in the 
        domain of the animation_variable.
    ani_speed : Number, optional
        Speed of the animation.
    angle : list, optional
        The view angle for 3D plots. Standart angle is [30, 30]
    ani_type : str, optional
        Specifies how the output sholud be animated. If no input is given, the method
        will try to use a fitting way, to show the data. Implemented types are:
            - 'line' for line animations, with a 1D-domain and output
            - 'surface_2D' for surface animation, with a 2D-domain
            - 'quiver_2D' for quiver/vector field animation, with a 2D-domain
            - 'contour_surface' for contour/colormaps, with a 2D-domain

    Returns
    -------
    plt.figure
        The figure handle of the created plot   
    animation.FuncAnimation
        The function that handles the animation  
    '''    
    assert isinstance(ani_sampler, AnimationSampler), \
        """The sampler has to be an AnimationSampler!"""
    animation_points, domain_points, outputs, out_shape = \
        _create_animation_data(model, ani_function, ani_sampler)
    ani_fun = _find_ani_function(ani_sampler, ani_type, out_shape)
    if ani_fun is not None:
        return ani_fun(outputs=outputs, ani_sampler=ani_sampler,
                       animation_points=animation_points, 
                       domain_points=domain_points,
                       angle=angle, ani_speed=ani_speed)
    else:
        raise NotImplementedError(f"""Animations for a {out_shape} 
                                      dimensional output are not implemented!'  
                                      Please specify the output to animate.""")


def _create_animation_data(model, ani_function, ani_sampler):
    # first create the plot points and evaluate the model
    animation_points = ani_sampler.sample_animation_points()
    domain_points = ani_sampler.sample_plot_domain_points(animation_points)
    # if the domain doesnt change with the animation the domain_points were
    # only created once. Therefore the model is easier to evaluate
    if ani_sampler.plot_domain_constant:
        return _animation_data_with_constant_domain(animation_points, domain_points, 
                                                    model, ani_function, ani_sampler)
    # else we have to copy the animations data depending on the number of 
    # domain points
    return _animation_data_with_changing_domain(animation_points, domain_points, 
                                                model, ani_function, ani_sampler)

def _animation_data_with_constant_domain(animation_points, domain_points, 
                                         model, ani_function, ani_sampler):
    inp_dict = ani_sampler._add_other_variables(domain_points)
    outputs = []
    ani_key = ani_sampler.animation_key
    # for each frame evaluate the model
    for i in range(ani_sampler.frame_number):
        repeat_point = animation_points[ani_key][i].repeat_interleave(len(ani_sampler))
        inp_dict[ani_key] = repeat_point.reshape(-1, 1)
        output = _evaluate_animation_function(model, ani_function, inp_dict)
        outputs.append(output)
    # get the output shape to determine the type of animation
    out_shape = _compute_output_shape(outputs[0])
    return animation_points, domain_points, outputs, out_shape


def _animation_data_with_changing_domain(animation_points, domain_points, 
                                         model, ani_function, ani_sampler):
    outputs = []
    ani_key = ani_sampler.animation_key
    # for each frame evaluate the model
    for i in range(ani_sampler.frame_number):
        # the number of points in the domain can change, if a density is used
        # therefore check always the current number of points
        current_length = ani_sampler._extract_tensor_len_from_dict(domain_points[i])
        ani_sampler.set_length(current_length)
        inp_dict = ani_sampler._add_other_variables(domain_points[i])
        repeat_point = animation_points[ani_key][i].repeat_interleave(len(ani_sampler))
        inp_dict[ani_key] = repeat_point.reshape(-1, 1)
        output = _evaluate_animation_function(model, ani_function, inp_dict)
        outputs.append(output)
    # get the output shape to determine the type of animation
    out_shape = _compute_output_shape(outputs[0])
    return animation_points, domain_points, outputs, out_shape


def _evaluate_animation_function(model, ani_function, inp_dict):
    model_out = model(inp_dict)
    inp = prepare_user_fun_input(ani_function, {**model_out, **inp_dict})
    output = ani_function(**inp)
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    return output


def _find_ani_function(ani_sampler, ani_type, out_shape):
    # check if a animation type is specified
    ani_types = {'line': animation_line, 'surface_2D': animation_surface2D,
                 'quiver_2D': animation_quiver_2D, 
                 'contour_surface': animation_contour_2D}
    ani_fun = ani_types.get(ani_type)
    # check ourself if we can animated the input and output dimension
    if ani_fun is None:
    # If only one output should be used we create a line/surface animation
        if out_shape == 1:
            ani_fun = _animation_for_one_output(ani_sampler.domain.dim)
        # If two outputs should be used we create a curve/quiver animation
        if out_shape == 2:
            ani_fun = _animation_for_two_outputs(ani_sampler.domain.dim)
    return ani_fun


def _animation_for_one_output(domain_dim):
    '''Handles animations if only one output of the model should be used.
    It will create a line or surface animation.
    '''
    # 2D animation (surface plot)
    if domain_dim == 2:
        return animation_surface2D
    # 1D animation (line plot):    
    elif domain_dim == 1:
        return animation_line
    else:
        raise NotImplementedError("""Can not animate 1D-output on given domain""")


def _animation_for_two_outputs(domain_dim):
    '''Handles animations if two outputs of the model should be used.
    It will create a curve or quiver animation.
    '''
    # animate quiver plot   
    if domain_dim == 2:
        return animation_quiver_2D
    else:
        raise NotImplementedError("""Can not animate 2D-output on given domain""")


def _extract_domain_points(input_dic, domain, length):
    # for the plot, the points need to be in a array not a dictionary
    # and are not allowed to be a tensor
    domain_points = np.zeros((length, domain.dim))
    current_dim = 0
    for vname in domain.space:
        v_dim = domain.space[vname]
        domain_points[:, current_dim:current_dim+v_dim] = \
            input_dic[vname].detach().cpu().numpy()
        current_dim += v_dim
    return domain_points


def animation_line(outputs, ani_sampler, animation_points, domain_points,
                   angle, ani_speed):
    '''Handels 1D animations, inputs are the same as animation().
    '''
    output_max, output_min, domain_bounds, ani_key, domain_name = \
        _compute_animation_params(outputs, ani_sampler, animation_points)
    # construct the figure handle and axis for the animation
    fig = plt.figure()
    ax = plt.axes(xlim=(domain_bounds[0], domain_bounds[1]),
                  ylim=(output_min, output_max))
    ax.set_xlabel(domain_name[0])
    ax.grid()
    line, = ax.plot([], [], lw=2)
    text_box = ax.text(0.05,0.95, '', bbox={'facecolor': 'w', 'pad': 5}, 
                       transform=ax.transAxes, va='top', ha='left')   
    # create the animation
    def animate(frame_number, outputs, line):
        if ani_sampler.plot_domain_constant:
            current_points = domain_points[domain_name[0]]
        else:
            current_points = domain_points[frame_number][domain_name[0]]
        # change line
        line.set_data(current_points.detach().cpu().numpy().flatten(),
                      outputs[frame_number].flatten())
        # change text-box data
        set_new_text(ani_sampler.dic_for_other_variables,
                     animation_points[ani_key][frame_number].item(),
                     ani_key,
                     text_box)
    
    ani = anim.FuncAnimation(fig, animate, frames=ani_sampler.frame_number, 
                             fargs=(outputs, line), interval=ani_speed)
    return fig, ani
    
    
def animation_surface2D(outputs, ani_sampler, animation_points, domain_points,
                        angle, ani_speed):
    '''Handels 2D animations, inputs are the same as animation().
    '''
    output_max, output_min, domain_bounds, ani_key, domain_name = \
        _compute_animation_params(outputs, ani_sampler, animation_points)
    # triangulate the domain once, if the it does not change
    triangulation = None
    if ani_sampler.plot_domain_constant:
        triangulation = _triangulate_for_animation(ani_sampler, domain_points)
    # construct the figure handle and axis for the animation
    fig, ax = _create_figure_and_axis(angle)
    _set_x_y_axis_data(domain_bounds, ax, domain_name)
    ax.set_zlim((output_min, output_max))
    text_box = ax.text2D(1.1, 0, '', bbox={'facecolor':'w', 'pad':5}, 
                         transform=ax.transAxes, va='top', ha='left')   
        
    # construct an auxiliary plot to get a fixed colorbar for the animation     
    surf = [ax.plot_surface(np.zeros((2, 2)),np.zeros((2, 2)),np.zeros((2, 2)), 
                            color='0.75', cmap=cm.jet, vmin=output_min,
                            vmax=output_max, antialiased=False)]
    plt.colorbar(surf[0], shrink=0.5, aspect=10, pad=0.1) 

    # create the animation
    def animate(frame_number, outputs, surf, triangulation):
        surf[0].remove() # remove old surface
        current_ani = animation_points[ani_key][frame_number]
        # have to create a new triangulation, if the domain changes
        if not ani_sampler.plot_domain_constant:
            triangulation = \
                _triangulate_for_animation(ani_sampler, domain_points[frame_number], 
                                           {ani_key: current_ani})
        surf[0] = ax.plot_trisurf(triangulation, outputs[frame_number].flatten(),
                                  color='0.75', cmap=cm.jet, 
                                  vmin=output_min, vmax=output_max, antialiased=False)
        _update_text_box(current_ani, ani_key, ani_sampler, text_box)

    ani = anim.FuncAnimation(fig, animate, frames=ani_sampler.frame_number, 
                             fargs=(outputs, surf, triangulation), interval=ani_speed)

    return fig, ani


def animation_quiver_2D(outputs, ani_sampler, animation_points, domain_points,
                        angle, ani_speed):
    '''Handles quiver animations in 2D
    '''
    _, _, domain_bounds, ani_key, domain_names = \
        _compute_animation_params(outputs, ani_sampler, animation_points)
    # for a consistent colors we compute the norm and scale the values
    color = []
    max_color = []
    norm = colors.Normalize()
    for i in range(ani_sampler.frame_number):
        color.append(np.linalg.norm(outputs[i], axis=-1))
        max_color.append(np.amax(color))
    norm.autoscale(color)
    j = np.argmax(max_color)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    _set_x_y_axis_data(domain_bounds, ax, domain_names)
    text_box = ax.text(1.25, 0.5, '', bbox={'facecolor': 'w', 'pad': 5},
                       transform=ax.transAxes)
    # helper quiver plot to get a fixed colorbar and a constant scaling
    current_length = ani_sampler._extract_tensor_len_from_dict(domain_points)
    numpy_domain = _extract_domain_points(domain_points, ani_sampler.domain,
                                          current_length)
    quiver = ax.quiver(numpy_domain[:, 0], numpy_domain[:, 1], 
                       outputs[j][:, 0], outputs[j][:, 1],
                       color=cm.jet(norm(color[j])),
                       scale=None, angles='xy', units='xy')
    sm = cm.ScalarMappable(cmap=cm.jet, norm=norm)
    quiver._init() # to fix the arrow scale
    plt.colorbar(sm)

    # create the animation
    def animate(frame_number, outputs, quiver):
        # set new coords. of arrow head and color
        print(quiver)
        quiver.remove()
        quiver = ax.quiver(numpy_domain[:, 0], numpy_domain[:, 1], 
                           outputs[frame_number][:, 0], outputs[frame_number][:, 1],
                           color=cm.jet(norm(color[frame_number])),
                           scale=None, angles='xy', units='xy')
        # get new point auf the animation variable and set text
        current_ani = animation_points[ani_key][frame_number]
        text_dict = {ani_key: current_ani.item(), 
                     **ani_sampler.dic_for_other_variables}
        info_string = _create_info_text(text_dict)  
        text_box.set_text(info_string)

    ani = anim.FuncAnimation(fig, animate, frames=ani_sampler.frame_number, 
                             fargs=(outputs, quiver), interval=ani_speed)

    return fig, ani


def animation_contour_2D(outputs, ani_sampler, animation_points, domain_points,
                         angle, ani_speed):
    '''Handles colormap animations in 2D
    '''
    output_max, output_min, domain_bounds, ani_key, domain_names = \
        _compute_animation_params(outputs, ani_sampler, animation_points)
    # Create the plot
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.grid()
    _set_x_y_axis_data(domain_bounds, ax, domain_names)
    text_box = ax.text(1.3, 0.5, '', bbox={'facecolor': 'w', 'pad': 5},
                       transform=ax.transAxes)
    # triangulate the domain once, if the it does not change
    triangulation = None
    if ani_sampler.plot_domain_constant:
        triangulation = _triangulate_for_animation(ani_sampler, domain_points)
    # helper plot for fixed colorbar
    con = [ax.scatter([0, 0], [0, 1], c=[output_min, output_max],
                      vmin=output_min, vmax=output_max, cmap=cm.jet)]
    plt.colorbar(con[0])
    con[0].remove()
    # create the animation
    def animate(frame_number, outputs, con, triangulation):
        current_ani = animation_points[ani_key][frame_number]
        # remove old contour
        if isinstance(con[0], plt_tri.TriContourSet):
            for tp in con[0].collections:
                tp.remove()
        # have to create a new triangulation, if the domain changes
        if not ani_sampler.plot_domain_constant:
            triangulation = \
                _triangulate_for_animation(ani_sampler, domain_points[frame_number], 
                                           {ani_key: current_ani})
        # set new contour
        con[0] = ax.tricontourf(triangulation, outputs[frame_number].flatten(),
                                100, cmap=cm.jet, vmin=output_min, vmax=output_max)
        # get new point auf the animation variable and set text
        _update_text_box(current_ani, ani_key, ani_sampler, text_box)
    
    ani = anim.FuncAnimation(fig, animate, frames=ani_sampler.frame_number, 
                             fargs=(outputs, con, triangulation), interval=ani_speed)

    return fig, ani


def _compute_animation_params(outputs, ani_sampler, animation_points):
    output_max, output_min = _get_max_min(outputs)
    domain_bounds = ani_sampler.domain.bounding_box(**animation_points)
    ani_key = ani_sampler.animation_key
    domain_name = list(ani_sampler.domain.space.keys())
    return output_max,output_min,domain_bounds,ani_key,domain_name


def _get_max_min(points):
    '''Returns the max and min value over all points.
    Needed to get a fixed y-(or z)axis.
    '''
    max_pro_output = []
    min_pro_output = []
    for p in points:
        max_pro_output.append(np.amax(p))
        min_pro_output.append(np.amin(p))
    return np.amax(max_pro_output), np.amin(min_pro_output)


def _set_x_y_axis_data(bounds, ax, domain_varibales):
    # set the border and add some margin
    width = bounds[1] - bounds[0]
    height = bounds[3] - bounds[2]
    scale_x = 0.05*width
    scale_y = 0.05*height
    ax.set_xlim((bounds[0]-scale_x, bounds[1]+scale_x))
    ax.set_ylim((bounds[2]-scale_y, bounds[3]+scale_y))
    if len(domain_varibales) == 1:
        ax.set_xlabel(domain_varibales[0] + '_1')
        ax.set_ylabel(domain_varibales[0] + '_2')
    else:
        ax.set_xlabel(domain_varibales[0])
        ax.set_ylabel(domain_varibales[1])  


def _triangulate_for_animation(ani_sampler, domain_points, ani_point={}):
    current_length = ani_sampler._extract_tensor_len_from_dict(domain_points)
    numpy_domain = _extract_domain_points(domain_points, ani_sampler.domain,
                                          current_length)
    evaluated_domain = ani_sampler.domain(**ani_point)
    triangulation = _triangulation_of_domain(evaluated_domain, numpy_domain)
    return triangulation


def set_new_text(variable_dic, new_ani_point, ani_name, text_box):
    variable_dic[ani_name] = new_ani_point
    info_string = _create_info_text(variable_dic)  
    text_box.set_text(info_string)


def _update_text_box(current_ani, ani_key, ani_sampler, text_box):
    # get new point auf the animation variable and set text
    text_dict = {ani_key: current_ani.item(), 
                 **ani_sampler.dic_for_other_variables}
    info_string = _create_info_text(text_dict)  
    text_box.set_text(info_string)